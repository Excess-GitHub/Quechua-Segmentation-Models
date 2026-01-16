"""
Neural network architectures for Quechua morphological segmentation.

Includes:
- BiLSTM boundary tagger (baseline)
- BiLSTM-CRF (with conditional random field)
- Transformer Seq2Seq transducer
- Decision Tree prior
- HMM (suffix-informed) prior
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ==============================================================================
# BiLSTM Boundary Tagger
# ==============================================================================

class BiLSTMBoundary(nn.Module):
    """
    Bidirectional LSTM for grapheme-level boundary prediction.
    
    Architecture:
        Embedding -> BiLSTM -> Dropout -> Linear -> Sigmoid
        
    Outputs probability of morpheme boundary at each position.
    """
    
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size * 2, 1)
        
    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Token IDs [batch, seq_len]
            lengths: Actual sequence lengths [batch]
            
        Returns:
            Boundary logits [batch, seq_len]
        """
        emb = self.emb(x)
        
        # Pack for efficient LSTM processing
        packed = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        
        out = self.dropout(out)
        logits = self.out(out).squeeze(-1)
        
        return logits


class BiLSTMWithPrior(nn.Module):
    """
    BiLSTM boundary tagger with external prior integration.
    
    Combines neural network predictions with linguistic priors (DT or HMM)
    via logit-level fusion.
    
    Architecture:
        Neural logits + α * log_odds(prior) -> Sigmoid
    """
    
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        prior_alpha: float = 1.0,
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.bilstm = BiLSTMBoundary(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=padding_idx
        )
        
        # Learnable weight for prior integration
        self.alpha = nn.Parameter(torch.tensor(prior_alpha))
        
    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        prior_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional prior integration.
        
        Args:
            x: Token IDs [batch, seq_len]
            lengths: Actual sequence lengths [batch]
            prior_probs: Prior boundary probabilities [batch, seq_len]
            
        Returns:
            Boundary logits [batch, seq_len]
        """
        logits = self.bilstm(x, lengths)
        
        if prior_probs is not None:
            # Convert prior probs to log-odds and fuse
            eps = 1e-6
            prior_probs = prior_probs.clamp(eps, 1 - eps)
            prior_logits = torch.log(prior_probs / (1 - prior_probs))
            logits = logits + self.alpha * prior_logits
            
        return logits


# ==============================================================================
# BiLSTM-CRF
# ==============================================================================

class BiLSTMCRF(nn.Module):
    """
    BiLSTM with Conditional Random Field for sequence labeling.
    
    The CRF layer models transition dependencies between boundary labels,
    using Viterbi decoding for inference.
    """
    
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_labels: int = 2,
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_size * 2, num_labels)
        
        # CRF transition parameters
        self.num_labels = num_labels
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels))
        self.start_transitions = nn.Parameter(torch.randn(num_labels))
        self.end_transitions = nn.Parameter(torch.randn(num_labels))
        
    def _get_emissions(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """Get emission scores from BiLSTM."""
        emb = self.emb(x)
        packed = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.dropout(out)
        emissions = self.hidden2tag(out)
        return emissions
    
    def _forward_algorithm(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute log partition function using forward algorithm."""
        batch_size, seq_len, num_tags = emissions.shape
        
        # Start with start transitions
        score = self.start_transitions + emissions[:, 0]
        
        for i in range(1, seq_len):
            # Broadcast for transition scores
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)
            
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            
            # Apply mask
            score = torch.where(mask[:, i].unsqueeze(1), next_score, score)
        
        # Add end transitions
        score = score + self.end_transitions
        return torch.logsumexp(score, dim=1)
    
    def _score_sentence(
        self,
        emissions: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute score for a given label sequence."""
        batch_size, seq_len, _ = emissions.shape
        
        # Emission scores
        score = self.start_transitions[labels[:, 0]]
        score += emissions[torch.arange(batch_size), 0, labels[:, 0]]
        
        for i in range(1, seq_len):
            # Transition + emission scores
            score += self.transitions[labels[:, i-1], labels[:, i]] * mask[:, i].float()
            score += emissions[torch.arange(batch_size), i, labels[:, i]] * mask[:, i].float()
        
        # Get last valid position for each sequence
        seq_ends = mask.long().sum(dim=1) - 1
        last_tags = labels[torch.arange(batch_size), seq_ends]
        score += self.end_transitions[last_tags]
        
        return score
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ):
        """
        Forward pass.
        
        Args:
            x: Token IDs [batch, seq_len]
            lengths: Actual sequence lengths [batch]
            labels: Ground truth labels [batch, seq_len] (for training)
            mask: Attention mask [batch, seq_len]
            
        Returns:
            If labels provided: negative log-likelihood loss
            Otherwise: decoded label sequences (list of lists)
        """
        emissions = self._get_emissions(x, lengths)
        
        if mask is None:
            mask = x != 0  # Assume 0 is padding
        
        if labels is not None:
            # Training: compute NLL loss
            forward_score = self._forward_algorithm(emissions, mask)
            gold_score = self._score_sentence(emissions, labels, mask)
            return (forward_score - gold_score).mean()
        else:
            # Inference: Viterbi decoding
            return self._viterbi_decode(emissions, mask)
    
    def _viterbi_decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor
    ) -> List[List[int]]:
        """Viterbi decoding for best label sequence."""
        batch_size, seq_len, num_tags = emissions.shape
        
        # Initialize
        score = self.start_transitions + emissions[:, 0]
        history = []
        
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)
            
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score, indices = next_score.max(dim=1)
            
            score = torch.where(mask[:, i].unsqueeze(1), next_score, score)
            history.append(indices)
        
        # Add end transitions
        score = score + self.end_transitions
        
        # Backtrack
        best_scores, best_tags = score.max(dim=1)
        
        best_paths = [[tag.item()] for tag in best_tags]
        
        for hist in reversed(history):
            best_tags = hist[torch.arange(batch_size), best_tags]
            for i, tag in enumerate(best_tags):
                best_paths[i].insert(0, tag.item())
        
        # Truncate to actual lengths
        seq_lens = mask.long().sum(dim=1).tolist()
        best_paths = [path[:length] for path, length in zip(best_paths, seq_lens)]
        
        return best_paths


# ==============================================================================
# Transformer Seq2Seq
# ==============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerSegmenter(nn.Module):
    """
    Transformer encoder-decoder for morphological segmentation.
    
    Input: character sequence
    Output: morpheme sequence with '+' separators
    """
    
    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 128,
        pad_idx: int = 0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Embeddings
        self.input_emb = nn.Embedding(input_vocab_size, d_model, padding_idx=pad_idx)
        self.output_emb = nn.Embedding(output_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.fc_out = nn.Linear(d_model, output_vocab_size)
        
        # Scale factor for embeddings
        self.scale = math.sqrt(d_model)
        
    def generate_square_subsequent_mask(self, sz: int, device) -> torch.Tensor:
        """Generate causal mask for autoregressive decoding."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            src: Source sequence [batch, src_len]
            tgt: Target sequence [batch, tgt_len]
            src_key_padding_mask: Mask for source padding
            tgt_key_padding_mask: Mask for target padding
            
        Returns:
            Output logits [batch, tgt_len, output_vocab_size]
        """
        # Embeddings with positional encoding
        src_emb = self.pos_encoder(self.input_emb(src) * self.scale)
        tgt_emb = self.pos_encoder(self.output_emb(tgt) * self.scale)
        
        # Causal mask for decoder
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), tgt.device)
        
        # Transformer
        out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        return self.fc_out(out)
    
    def greedy_decode(
        self,
        src: torch.Tensor,
        max_len: int,
        sos_idx: int,
        eos_idx: int
    ) -> torch.Tensor:
        """
        Greedy decoding for inference.
        
        Args:
            src: Source sequence [batch, src_len]
            max_len: Maximum output length
            sos_idx: Start-of-sequence token index
            eos_idx: End-of-sequence token index
            
        Returns:
            Decoded sequence [batch, decoded_len]
        """
        batch_size = src.size(0)
        device = src.device
        
        # Encode source
        src_emb = self.pos_encoder(self.input_emb(src) * self.scale)
        memory = self.transformer.encoder(src_emb)
        
        # Initialize with SOS
        ys = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            tgt_emb = self.pos_encoder(self.output_emb(ys) * self.scale)
            tgt_mask = self.generate_square_subsequent_mask(ys.size(1), device)
            
            out = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            logits = self.fc_out(out[:, -1])
            next_token = logits.argmax(dim=-1, keepdim=True)
            
            ys = torch.cat([ys, next_token], dim=1)
            
            # Stop if all sequences have generated EOS
            if (next_token == eos_idx).all():
                break
        
        return ys


# ==============================================================================
# Decision Tree Prior
# ==============================================================================

class DecisionTreePrior:
    """
    Decision Tree classifier for local boundary probability estimation.
    
    Features extracted at each position:
    - Context tokens (left/right neighbors)
    - CV pattern of neighbors
    - Character identity features
    """
    
    def __init__(self, max_depth: int = 6):
        self.max_depth = max_depth
        self.clf = None
        self.vec = None
        self._fitted = False
        
    def _extract_features(
        self,
        tokens: List[str],
        position: int
    ) -> Dict:
        """Extract features for a single position."""
        n = len(tokens)
        vowels = set("aeiou")
        
        # Get context
        l1 = tokens[position] if position < n else ""
        l2 = tokens[position - 1] if position >= 1 else ""
        r1 = tokens[position + 1] if position + 1 < n else ""
        r2 = tokens[position + 2] if position + 2 < n else ""
        
        # CV pattern
        def cv(t):
            return "V" if t.lower() in vowels else "C" if t else ""
        
        features = {
            "L1": l1,
            "L2": l2,
            "R1": r1,
            "R2": r2,
            "L1_cv": cv(l1),
            "L2_cv": cv(l2),
            "R1_cv": cv(r1),
            "R2_cv": cv(r2),
            "last_char_L1": l1[-1:] if l1 else "",
            "first_char_R1": r1[:1] if r1 else "",
        }
        
        return features
    
    def fit(
        self,
        words_tokens: List[List[str]],
        boundary_labels: List[List[int]]
    ):
        """
        Train the Decision Tree classifier.
        
        Args:
            words_tokens: List of token sequences
            boundary_labels: List of boundary label sequences
        """
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.tree import DecisionTreeClassifier
        
        X_dicts = []
        y = []
        
        for tokens, labels in zip(words_tokens, boundary_labels):
            for i in range(len(tokens) - 1):  # No boundary prediction at last position
                X_dicts.append(self._extract_features(tokens, i))
                y.append(labels[i])
        
        self.vec = DictVectorizer(sparse=False)
        X = self.vec.fit_transform(X_dicts)
        
        self.clf = DecisionTreeClassifier(max_depth=self.max_depth, random_state=42)
        self.clf.fit(X, y)
        self._fitted = True
        
        print(f"DT prior: {self.clf.tree_.node_count} nodes, depth={self.clf.get_depth()}")
        
    def predict_probs(self, tokens: List[str]) -> List[float]:
        """
        Predict boundary probabilities for all positions.
        
        Args:
            tokens: Grapheme token sequence
            
        Returns:
            List of boundary probabilities [0, 1]
        """
        assert self._fitted, "Call fit() first"
        
        if len(tokens) <= 1:
            return [0.0] * len(tokens)
        
        probs = []
        for i in range(len(tokens)):
            if i == len(tokens) - 1:
                probs.append(0.0)  # No boundary at end
            else:
                feats = self._extract_features(tokens, i)
                X = self.vec.transform([feats])
                prob = self.clf.predict_proba(X)[0]
                probs.append(prob[1] if len(prob) > 1 else 0.0)
        
        return probs


# ==============================================================================
# HMM Suffix Prior
# ==============================================================================

class HMMSuffixPrior:
    """
    Hidden Markov Model prior using suffix vocabulary.
    
    Uses forward-backward algorithm to estimate boundary probabilities
    based on learned suffix log-probabilities.
    """
    
    def __init__(self, max_suffix_len: int = 8, unk_penalty: float = -11.53):
        self.max_suffix_len = max_suffix_len
        self.unk_penalty = unk_penalty
        self.suffix_log_probs = {}
        self.suffixes = set()
        self._fitted = False
        
    def fit(self, morph_splits: List[List[str]]):
        """
        Train the HMM prior from gold segmentations.
        
        Extracts all non-root morphemes as the suffix vocabulary
        and computes smoothed log-probabilities.
        
        Args:
            morph_splits: List of morpheme lists (first is root, rest are suffixes)
        """
        suffix_counts = Counter()
        
        for morphs in morph_splits:
            if len(morphs) > 1:
                for suffix in morphs[1:]:
                    s = suffix.lower()
                    if len(s) <= self.max_suffix_len:
                        suffix_counts[s] += 1
        
        # Compute smoothed log-probabilities
        total = sum(suffix_counts.values())
        vocab_size = len(suffix_counts)
        
        self.suffix_log_probs = {}
        for suffix, count in suffix_counts.items():
            self.suffix_log_probs[suffix] = np.log((count + 1) / (total + vocab_size))
        
        self.suffixes = set(suffix_counts.keys())
        self.unk_penalty = np.log(1 / (total + vocab_size)) - 2  # Extra penalty
        self._fitted = True
        
        print(f"HMM: {len(self.suffixes)} suffixes, max len {self.max_suffix_len}, unk penalty {self.unk_penalty:.2f}")
        
    def _log_prob(self, substring: str) -> float:
        """Get log-probability of a substring as a suffix."""
        s = substring.lower()
        return self.suffix_log_probs.get(s, self.unk_penalty)
    
    def predict_probs(self, tokens: List[str]) -> List[float]:
        """
        Compute boundary probabilities using forward-backward.
        
        Args:
            tokens: Grapheme token sequence
            
        Returns:
            List of boundary probabilities
        """
        assert self._fitted, "Call fit() first"
        
        n = len(tokens)
        if n <= 1:
            return [0.0] * n
        
        word = "".join(tokens)
        word_len = len(word)
        
        # Build character-to-token mapping
        char_to_tok = []
        for i, tok in enumerate(tokens):
            char_to_tok.extend([i] * len(tok))
        
        # Forward pass (in log space)
        NEG_INF = -1e9
        alpha = [NEG_INF] * (word_len + 1)
        alpha[0] = 0.0
        
        for i in range(1, word_len + 1):
            for j in range(max(0, i - self.max_suffix_len * 2), i):
                substring = word[j:i]
                if len(substring) <= self.max_suffix_len:
                    log_p = self._log_prob(substring)
                    alpha[i] = np.logaddexp(alpha[i], alpha[j] + log_p)
        
        # Backward pass (in log space)
        beta = [NEG_INF] * (word_len + 1)
        beta[word_len] = 0.0
        
        for i in range(word_len - 1, -1, -1):
            for j in range(i + 1, min(word_len + 1, i + self.max_suffix_len * 2 + 1)):
                substring = word[i:j]
                if len(substring) <= self.max_suffix_len:
                    log_p = self._log_prob(substring)
                    beta[i] = np.logaddexp(beta[i], beta[j] + log_p)
        
        # Compute boundary probabilities at token boundaries
        log_Z = alpha[word_len]
        probs = []
        
        char_pos = 0
        for i in range(n):
            char_pos += len(tokens[i])
            if i == n - 1:
                probs.append(0.0)  # No boundary at end
            else:
                # P(boundary at char_pos) = sum over all paths with boundary here
                if alpha[char_pos] > NEG_INF / 2 and beta[char_pos] > NEG_INF / 2:
                    log_boundary = alpha[char_pos] + beta[char_pos] - log_Z
                    prob = np.exp(np.clip(log_boundary, -20, 0))
                    probs.append(min(prob, 1.0))
                else:
                    probs.append(0.0)
        
        return probs


# ==============================================================================
# Rejection Filter
# ==============================================================================

class SuffixRejectionFilter:
    """
    Post-processing filter that rejects invalid segmentations.
    
    If any predicted non-root segment is not in the known suffix vocabulary,
    the entire segmentation is rejected and the word is returned unsegmented.
    """
    
    def __init__(self, suffix_set: set):
        self.suffix_set = {s.lower() for s in suffix_set}
        
    def validate(self, segments: List[str]) -> bool:
        """
        Check if all non-root segments are valid suffixes.
        
        Args:
            segments: List of predicted morpheme segments
            
        Returns:
            True if valid, False if should be rejected
        """
        if len(segments) <= 1:
            return True
        
        for seg in segments[1:]:  # Skip root
            if seg.lower() not in self.suffix_set:
                return False
        
        return True
    
    def filter(self, word: str, segments: List[str]) -> List[str]:
        """
        Apply filter, returning unsegmented word if invalid.
        
        Args:
            word: Original word
            segments: Predicted segments
            
        Returns:
            Filtered segments (or [word] if rejected)
        """
        if self.validate(segments):
            return segments
        return [word]
