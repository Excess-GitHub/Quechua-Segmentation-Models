#!/usr/bin/env python3
"""
Script to generate a refactored version of the segmenter-morfessor notebook.
Uses nbformat to construct the notebook programmatically.
"""

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell


def create_notebook():
    """Create and return a new notebook object."""
    nb = new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    }
    return nb


def add_header_section(nb):
    """Add title and overview markdown."""
    nb.cells.append(new_markdown_cell(
        "# Segmenter-Morfessor: BiLSTM with Morfessor Ensemble Priors\n\n"
        "Grapheme-level BiLSTM for Quechua morphological segmentation using "
        "Morfessor ensemble boundary probabilities as an additional feature channel."
    ))


def add_imports_section(nb):
    """Add all imports in a single organized cell."""
    imports_code = '''# Core libraries
import ast
import os
import json
import hashlib
import pickle
import random
import unicodedata
import string
from dataclasses import dataclass
from typing import List, Dict, Tuple

# Data handling
import numpy as np
import pandas as pd
import regex as re

# ML & DL
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

# Morfessor
import morfessor'''
    nb.cells.append(new_code_cell(imports_code))


def add_config_section(nb):
    """Add configuration constants."""
    config_code = '''# Paths
DATA_FOLDER = "data"
MODEL_NAME = "segmenter-morfessor"
MODELS_FOLDER = f"models_{MODEL_NAME}"
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# Special tokens
PAD, UNK = "<PAD>", "<UNK>"

# Text normalization constants
APOSTROPHE_CHARS = {"'", "'", "ʼ", "‛", "`"}
STD_APOS = "\\u02BC"
_EXTRA_PUNCT = "±，"'"'"
_DELETE = str.maketrans("", "", string.punctuation + _EXTRA_PUNCT)

# Quechua multigraphs
QUECHUA_MULTIGRAPHS = [
    "ch" + STD_APOS, "k" + STD_APOS, "p" + STD_APOS, "q" + STD_APOS, "t" + STD_APOS,
    "ch", "ph", "qh", "kh", "ll", "rr", "sh",
]
MG_SET = set(QUECHUA_MULTIGRAPHS)
MAX_MG = max((len(mg) for mg in QUECHUA_MULTIGRAPHS), default=1)'''
    nb.cells.append(new_code_cell(config_code))


def add_text_normalization_section(nb):
    """Add text normalization and grapheme tokenization."""
    
    norm_code = '''def normalize_text(s: str) -> str:
    """Normalize text: NFC compose, lowercase, unify apostrophes, strip punctuation."""
    s = unicodedata.normalize("NFC", str(s)).lower()
    s = "".join(STD_APOS if ch in APOSTROPHE_CHARS else ch for ch in s)
    s = s.translate(_DELETE).strip()
    return s

def to_graphemes_quechua(s: str) -> list[str]:
    """Greedy longest-match multigraph fusion; fallback to Unicode grapheme clusters."""
    s = normalize_text(s)
    tokens, i, n = [], 0, len(s)
    while i < n:
        match = None
        for L in range(MAX_MG, 1, -1):
            if i + L <= n:
                cand = s[i:i+L]
                if cand in MG_SET:
                    match = cand
                    break
        if match:
            tokens.append(match)
            i += len(match)
        else:
            m = re.match(r"\\X", s[i:])
            g = m.group(0)
            tokens.append(g)
            i += len(g)
    return tokens'''
    nb.cells.append(new_code_cell(norm_code))


def add_data_loading_section(nb):
    """Add data loading and preprocessing."""
    
    data_loading = '''# Load gold standard data
print("loading gold data...")
gold_df = pd.read_parquet(os.path.join(DATA_FOLDER, "Sue_kalt.parquet"))
gold_df['Word'] = gold_df['word']
gold_df['morph'] = gold_df['morph'].str.replace('-', ' ')
gold_df['Morph_split_str'] = gold_df['morph']
gold_df['Morph_split'] = gold_df['morph'].str.split(' ')
gold_df = gold_df[['Word', 'Morph_split', 'Morph_split_str']]
gold_df = gold_df.drop_duplicates(subset=['Word']).reset_index(drop=True)
gold_df = gold_df.dropna(subset=['Word'])
print(f"got {len(gold_df):,} gold examples")'''
    nb.cells.append(new_code_cell(data_loading))
    
    tokenization = '''# Tokenize words and morphemes to graphemes
gold_df['token_seq'] = gold_df['Word'].apply(lambda w: to_graphemes_quechua(w))
gold_df['morph_token_splits'] = gold_df['Morph_split'].apply(
    lambda var: [to_graphemes_quechua(m) for m in var]
)'''
    nb.cells.append(new_code_cell(tokenization))
    
    boundary_labels = '''def get_boundary_labels_tokens(tokens: list[str], morph_tokens: list[list[str]]) -> list[int]:
    """Generate binary boundary labels for a word given its morpheme token splits."""
    labels = [0] * len(tokens)
    idx = 0
    for mt in morph_tokens[:-1]:
        idx += len(mt)
        if 0 < idx <= len(tokens):
            labels[idx-1] = 1
    return labels

gold_df['boundary_labels'] = gold_df.apply(
    lambda row: get_boundary_labels_tokens(row['token_seq'], row['morph_token_splits']),
    axis=1
)

gold_df['num_morphemes'] = gold_df['Morph_split'].apply(len)
gold_df['word_len_tokens'] = gold_df['token_seq'].apply(len)
gold_df['char_seq'] = gold_df['token_seq']  # compatibility alias'''
    nb.cells.append(new_code_cell(boundary_labels))


def add_morfessor_section(nb):
    """Add Morfessor ensemble featurizer."""
    
    morfessor_code = '''@dataclass
class MorfessorConfig:
    n_models: int = 5
    seed_base: int = 123
    corpus_min_count: int = 1
    lowercase: bool = True

class MorfessorBoundaryFeaturizer:
    """Morfessor ensemble that provides boundary probabilities as features."""
    def __init__(self, cfg: MorfessorConfig):
        self.cfg = cfg
        self.models = []
        self._fitted = False

    def _build_model(self, seed: int):
        """Build a single Morfessor Baseline model."""
        io = morfessor.MorfessorIO()
        model = morfessor.BaselineModel()
        random.seed(seed)
        np.random.seed(seed)
        return model

    def fit(self, words: List[str]):
        """Train N models with different random shuffles/seeds."""
        words = [w.lower() if self.cfg.lowercase else w for w in words]
        uniq = list(set(words))
        rng = np.random.default_rng(self.cfg.seed_base)
        self.models = []

        for i in range(self.cfg.n_models):
            seed = self.cfg.seed_base + i
            model = self._build_model(seed)
            shuffled = uniq.copy()
            rng.shuffle(shuffled)
            train_data = [(1, w) for w in shuffled]
            model.load_data(train_data)
            model.train_batch()
            self.models.append(model)

        self._fitted = True

    @staticmethod
    def _boundaries_from_segments(word: str, segments: List[str]) -> List[int]:
        """Return hard boundary vector over raw characters."""
        b = [0] * len(word)
        pos = 0
        for seg_i, seg in enumerate(segments[:-1]):
            pos += len(seg)
            if 0 <= pos - 1 < len(word):
                b[pos - 1] = 1
        return b

    @staticmethod
    def _char_to_token_boundaries(word: str, tokens: List[str], char_bound: List[int]) -> List[float]:
        """Map char-level boundaries to tokenization."""
        rebuilt = "".join(tokens)
        if len(rebuilt) != len(char_bound):
            char_bound = (char_bound[:len(rebuilt)] + [0]*(len(rebuilt) - len(char_bound)))[:len(rebuilt)]

        token_ends = []
        p = 0
        for tok in tokens:
            p += len(tok)
            token_ends.append(p - 1)

        probs = [0.0] * len(tokens)
        for t_i, char_end in enumerate(token_ends[:-1]):
            probs[t_i] = float(char_bound[char_end])
        return probs

    def boundary_probs_for_word(self, word: str, tokens: List[str]) -> List[float]:
        """Get boundary probabilities for a single word from ensemble."""
        assert self._fitted, "call fit() first"
        word_ = word.lower() if self.cfg.lowercase else word

        per_model_probs = []
        for m in self.models:
            segs, _ = m.viterbi_segment(word_)
            char_b = self._boundaries_from_segments(word_, segs)
            per_model_probs.append(self._char_to_token_boundaries(word_, tokens, char_b))

        if not per_model_probs:
            return [0.0] * len(tokens)
        probs = np.mean(np.array(per_model_probs), axis=0).tolist()
        return probs

    def boundary_probs_for_words(self, words: List[str], tok_lists: List[List[str]]) -> List[List[float]]:
        """Get boundary probabilities for multiple words."""
        return [self.boundary_probs_for_word(w, toks) for w, toks in zip(words, tok_lists)]'''
    nb.cells.append(new_code_cell(morfessor_code))


def add_vocabulary_section(nb):
    """Add vocabulary construction."""
    
    vocab_code = '''def build_vocab(seqs: List[List[str]]):
    """Build vocabulary from grapheme token sequences."""
    toks = {t for seq in seqs for t in seq}
    itos = [PAD, UNK] + sorted(toks)
    stoi = {t: i for i, t in enumerate(itos)}
    return stoi, itos

stoi, itos = build_vocab(gold_df["char_seq"].tolist())
print(f"vocab size: {len(itos)} graphemes")

def encode(seq: List[str]) -> List[int]:
    """Convert grapheme sequence to integer IDs."""
    return [stoi.get(t, stoi[UNK]) for t in seq]

def encode_labels(labels: List[int]) -> List[int]:
    """Labels are already 0/1."""
    return labels'''
    nb.cells.append(new_code_cell(vocab_code))


def add_morfessor_io_section(nb):
    """Add Morfessor save/load functions."""
    
    morf_io = '''def generate_morfessor_id(n_models, seed_base, lowercase):
    """Hash Morfessor config to get unique ID."""
    params_dict = {
        'n_models': n_models,
        'seed_base': seed_base,
        'lowercase': lowercase
    }
    params_str = json.dumps(params_dict, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()[:16]

def save_morfessor_ensemble(morf_featurizer, morfessor_id, models_folder=MODELS_FOLDER):
    """Save Morfessor ensemble models."""
    morfessor_dir = os.path.join(models_folder, f"morfessor_{morfessor_id}")
    os.makedirs(morfessor_dir, exist_ok=True)
    
    for i, model in enumerate(morf_featurizer.models):
        model_path = os.path.join(morfessor_dir, f"morfessor_model_{i}.pkl}")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    config_path = os.path.join(morfessor_dir, "morfessor_config.json")
    with open(config_path, "w") as f:
        json.dump({
            'n_models': morf_featurizer.cfg.n_models,
            'seed_base': morf_featurizer.cfg.seed_base,
            'lowercase': morf_featurizer.cfg.lowercase,
            'morfessor_id': morfessor_id
        }, f, indent=2)
    
    print(f"saved morfessor ensemble to {morfessor_dir}")
    return morfessor_dir

def load_morfessor_ensemble(morfessor_id, models_folder=MODELS_FOLDER):
    """Load Morfessor ensemble models."""
    morfessor_dir = os.path.join(models_folder, f"morfessor_{morfessor_id}")
    config_path = os.path.join(morfessor_dir, "morfessor_config.json")
    
    if not os.path.exists(config_path):
        return None
    
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    m_cfg = MorfessorConfig(
        n_models=config_data['n_models'],
        seed_base=config_data['seed_base'],
        lowercase=config_data['lowercase']
    )
    morf = MorfessorBoundaryFeaturizer(m_cfg)
    
    morf.models = []
    for i in range(m_cfg.n_models):
        model_path = os.path.join(morfessor_dir, f"morfessor_model_{i}.pkl")
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                morf.models.append(pickle.load(f))
        else:
            print(f"warning: morfessor model {i} not found at {model_path}")
            return None
    
    morf._fitted = True
    print(f"loaded morfessor ensemble from {morfessor_dir}")
    return morf'''
    nb.cells.append(new_code_cell(morf_io))


def add_train_morfessor_section(nb):
    """Add Morfessor training/loading cell."""
    
    train_morf = '''# Train or load Morfessor ensemble
m_cfg = MorfessorConfig(n_models=5, seed_base=123, lowercase=True)
morfessor_id = generate_morfessor_id(m_cfg.n_models, m_cfg.seed_base, m_cfg.lowercase)

print(f"looking for morfessor ensemble {morfessor_id}...")
morf = load_morfessor_ensemble(morfessor_id, models_folder=MODELS_FOLDER)

if morf is None:
    print(f"not found, training...")
    morf = MorfessorBoundaryFeaturizer(m_cfg)
    morf.fit(gold_df["Word"].tolist())
    save_morfessor_ensemble(morf, morfessor_id, models_folder=MODELS_FOLDER)
    print(f"morfessor ensemble training done! saved with ID: {morfessor_id}")
else:
    print(f"using existing morfessor ensemble (ID: {morfessor_id})")'''
    nb.cells.append(new_code_cell(train_morf))


def add_dataset_section(nb):
    """Add dataset and dataloader code."""
    
    dataset_code = '''class CharBoundaryDatasetMorf(torch.utils.data.Dataset):
    """Dataset with precomputed Morfessor boundary probabilities."""
    def __init__(self, df, morf_featurizer: MorfessorBoundaryFeaturizer, stoi):
        self.words = df["Word"].tolist()
        self.x = df["char_seq"].tolist()
        self.y = df["boundary_labels"].tolist()
        self.morf = morf_featurizer.boundary_probs_for_words(self.words, self.x)
        self.stoi = stoi

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.morf[idx]

def pad_batch_with_morf(batch, pad_id=0):
    """Collate function: pads sequences and Morfessor features."""
    seqs, labels, morf = zip(*batch)

    x_ids = [[stoi.get(t, stoi["<UNK>"]) for t in s] for s in seqs]
    y_ids = [lab for lab in labels]
    m_probs = [mp for mp in morf]

    lengths = [len(x) for x in x_ids]
    maxlen = max(lengths)

    x_pad = [xi + [pad_id] * (maxlen - len(xi)) for xi in x_ids]
    y_pad = [yi + [0] * (maxlen - len(yi)) for yi in y_ids]
    m_pad = [mi + [0.0] * (maxlen - len(mi)) for mi in m_probs]
    mask = [[1] * len(xi) + [0] * (maxlen - len(xi)) for xi in x_ids]

    return (
        torch.LongTensor(x_pad),
        torch.FloatTensor(y_pad),
        torch.BoolTensor(mask),
        torch.LongTensor(lengths),
        torch.FloatTensor(m_pad).unsqueeze(-1),
    )'''
    nb.cells.append(new_code_cell(dataset_code))
    
    split_code = '''# Train/validation split
rng = np.random.default_rng(42)
indices = np.arange(len(gold_df))
rng.shuffle(indices)
split = int(0.9 * len(indices))
train_idx, val_idx = indices[:split], indices[split:]

train_df = gold_df.iloc[train_idx].reset_index(drop=True)
val_df = gold_df.iloc[val_idx].reset_index(drop=True)

print(f"training: {len(train_df):,} samples")
print(f"validation: {len(val_df):,} samples")

train_ds = CharBoundaryDatasetMorf(train_df, morf, stoi)
val_ds = CharBoundaryDatasetMorf(val_df, morf, stoi)

BATCH_SIZE = 64
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_batch_with_morf)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_batch_with_morf)'''
    nb.cells.append(new_code_cell(split_code))


def add_model_section(nb):
    """Add BiLSTM model with Morfessor features."""
    
    model_code = '''class BiLSTMBoundaryWithMorf(nn.Module):
    """Bidirectional LSTM with Morfessor feature channel."""
    def __init__(self, vocab_size: int, emb_dim: int = 16, hidden_size: int = 16,
                 num_layers: int = 1, dropout: float = 0.1, freeze_emb: bool = False,
                 extra_feat_dim: int = 1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if freeze_emb:
            for p in self.emb.parameters():
                p.requires_grad = False

        self.in_dim = emb_dim + extra_feat_dim
        self.lstm = nn.LSTM(
            input_size=self.in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size * 2, 1)

    def forward(self, x_ids, lengths, extra_feats):
        """Forward pass: concatenate embeddings with Morfessor features."""
        emb = self.emb(x_ids)
        x_in = torch.cat([emb, extra_feats], -1)

        packed = nn.utils.rnn.pack_padded_sequence(
            x_in, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        out = self.dropout(out)
        logits = self.out(out).squeeze(-1)
        return logits'''
    nb.cells.append(new_code_cell(model_code))


def add_loss_section(nb):
    """Add loss function."""
    
    loss_code = '''def count_pos_neg(df_):
    """Count positive and negative examples for class weighting."""
    pos = sum(sum(lbls) for lbls in df_['boundary_labels'])
    total = sum(len(seq) for seq in df_['char_seq'])
    neg = total - pos
    return pos, neg

pos, neg = count_pos_neg(gold_df)
pos_weight_value = float(neg) / max(float(pos), 1.0)

def masked_bce_loss(logits, targets, mask):
    """Compute masked binary cross-entropy loss."""
    loss_fn = nn.BCEWithLogitsLoss(reduction="none",
                                   pos_weight=torch.tensor(pos_weight_value, device=logits.device))
    loss_per_token = loss_fn(logits, targets) * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return loss_per_token.sum() / denom'''
    nb.cells.append(new_code_cell(loss_code))


def add_metrics_section(nb):
    """Add evaluation metrics."""
    
    metrics_code = '''def boundary_f1(logits, targets, mask, threshold=0.5):
    """Compute precision, recall, and F1 for boundary prediction."""
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).long()
        t = targets.long()
        m = mask.long()

        tp = ((preds == 1) & (t == 1) & (m == 1)).sum().item()
        fp = ((preds == 1) & (t == 0) & (m == 1)).sum().item()
        fn = ((preds == 0) & (t == 1) & (m == 1)).sum().item()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1'''
    nb.cells.append(new_code_cell(metrics_code))


def add_inference_section(nb):
    """Add inference functions."""
    
    inference_code = '''def predict_boundaries_with_morf(words: List[str], model, stoi, morf_featurizer,
                                 threshold=0.5, device=device):
    """Predict boundary labels for a list of words using Morfessor features."""
    model.eval()
    token_lists = [to_graphemes_quechua(w) for w in words]

    x_ids = [[stoi.get(t, stoi["<UNK>"]) for t in toks] for toks in token_lists]
    lengths = [len(x) for x in x_ids]
    maxlen = max(lengths) if lengths else 0
    pad_id = stoi["<PAD>"]
    x_pad = [xi + [pad_id] * (maxlen - len(xi)) for xi in x_ids]
    mask = [[1] * len(xi) + [0] * (maxlen - len(xi)) for xi in x_ids]

    morf_probs = morf_featurizer.boundary_probs_for_words(words, token_lists)
    morf_pad = [mp + [0.0] * (maxlen - len(mp)) for mp in morf_probs]
    morf_feat = torch.FloatTensor(morf_pad).unsqueeze(-1)

    x = torch.LongTensor(x_pad).to(device)
    lengths_t = torch.LongTensor(lengths).to(device)
    mask_t = torch.BoolTensor(mask).to(device)
    morf_feat = morf_feat.to(device)

    with torch.no_grad():
        logits = model(x, lengths_t, morf_feat)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold) & mask_t

    out = []
    for i, L in enumerate(lengths):
        out.append(preds[i, :L].int().tolist())
    return out

def apply_boundaries_tokens(tokens: list[str], boundary_labels: List[int]) -> List[str]:
    """Reconstruct morphemes from token list and boundary labels."""
    segs, start = [], 0
    for i, b in enumerate(boundary_labels):
        if b == 1:
            segs.append("".join(tokens[start:i+1]))
            start = i + 1
    if start < len(tokens):
        segs.append("".join(tokens[start:]))
    return segs

def normalize_gold_variants(gold_variants):
    """Convert gold_variants to proper list format."""
    if gold_variants is None:
        return []
    if isinstance(gold_variants, np.ndarray):
        gold_variants = gold_variants.tolist()
    if isinstance(gold_variants, list):
        normalized = []
        for variant in gold_variants:
            if isinstance(variant, np.ndarray):
                normalized.append(variant.tolist())
            elif isinstance(variant, list):
                normalized.append([item.tolist() if isinstance(item, np.ndarray) else item for item in variant])
            else:
                normalized.append(variant)
        return normalized
    return []

def evaluate_accuracy_morf(df, model, stoi, morf_featurizer, device="cpu", threshold=0.5):
    """Accuracy = proportion of words where predicted segmentation == any gold variant."""
    all_words = df["Word"].tolist()
    all_gold = df["Gold"].tolist()

    all_boundaries = predict_boundaries_with_morf(
        all_words, model, stoi, morf_featurizer, threshold=threshold, device=device
    )

    correct = 0
    for word, gold_variants, boundary_labels in zip(all_words, all_gold, all_boundaries):
        gold_variants = normalize_gold_variants(gold_variants)
        toks = to_graphemes_quechua(word)
        predicted = apply_boundaries_tokens(toks, boundary_labels)
        if any(predicted == variant for variant in gold_variants):
            correct += 1
    return correct / len(all_words) if all_words else 0.0'''
    nb.cells.append(new_code_cell(inference_code))


def add_checkpointing_section(nb):
    """Add model checkpointing functions."""
    
    checkpoint_code = '''def generate_model_id(vocab_size, emb_dim, hidden_size, num_layers, dropout,
                      freeze_emb, extra_feat_dim, lr, weight_decay, morfessor_id, epochs):
    """Hash training params to get unique model ID."""
    params_dict = {
        'vocab_size': vocab_size,
        'emb_dim': emb_dim,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'dropout': dropout,
        'freeze_emb': freeze_emb,
        'extra_feat_dim': extra_feat_dim,
        'lr': lr,
        'weight_decay': weight_decay,
        'morfessor_id': morfessor_id,
        'epochs': epochs
    }
    params_str = json.dumps(params_dict, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()[:16]

def save_model_checkpoint(model, stoi, itos, model_id, models_folder=MODELS_FOLDER,
                         suffix="", best_metric_value=None):
    """Save BiLSTM model checkpoint."""
    model_dir = os.path.join(models_folder, f"model_{model_id}")
    os.makedirs(model_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(model_dir, f"bilstm_morfessor{suffix}.pt")
    checkpoint = {
        "model_state": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "model_id": model_id,
        "morfessor_id": morfessor_id
    }
    if best_metric_value is not None:
        checkpoint["best_metric_value"] = best_metric_value
    
    torch.save(checkpoint, checkpoint_path)
    print(f"saved checkpoint to {checkpoint_path}")
    return checkpoint_path

def load_model_checkpoint(model_id, models_folder=MODELS_FOLDER, suffix=""):
    """Load BiLSTM model checkpoint."""
    model_dir = os.path.join(models_folder, f"model_{model_id}")
    checkpoint_path = os.path.join(model_dir, f"bilstm_morfessor{suffix}.pt")
    
    if not os.path.exists(checkpoint_path):
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"loaded checkpoint from {checkpoint_path}")
    return checkpoint'''
    nb.cells.append(new_code_cell(checkpoint_code))


def add_model_config_section(nb):
    """Add model configuration and initialization."""
    
    config_code = '''# Model hyperparameters
VOCAB_SIZE = len(itos)
EMB_DIM = 64
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
FREEZE_EMB = True
EXTRA_FEAT_DIM = 1

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 20

model_id = generate_model_id(
    vocab_size=VOCAB_SIZE,
    emb_dim=EMB_DIM,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    freeze_emb=FREEZE_EMB,
    extra_feat_dim=EXTRA_FEAT_DIM,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    morfessor_id=morfessor_id,
    epochs=EPOCHS
)

print(f"model ID: {model_id}")
print(f"using morfessor ensemble ID: {morfessor_id}")

# Check if model already exists
print(f"looking for model {model_id}...")
checkpoint = load_model_checkpoint(model_id, models_folder=MODELS_FOLDER, suffix="_best_f1")

if checkpoint is not None:
    print(f"found it! loading checkpoint...")
    model_loaded = True
    saved_stoi = checkpoint["stoi"]
    saved_itos = checkpoint["itos"]
else:
    print(f"not found, will train new model")
    model_loaded = False
    saved_stoi = None
    saved_itos = None

# Create model
model = BiLSTMBoundaryWithMorf(
    vocab_size=VOCAB_SIZE,
    emb_dim=EMB_DIM,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    freeze_emb=FREEZE_EMB,
    extra_feat_dim=EXTRA_FEAT_DIM
).to(device)

# If model was loaded, restore its state
if model_loaded:
    model.load_state_dict(checkpoint["model_state"])
    print("model state restored from checkpoint")
    if saved_stoi == stoi and saved_itos == itos:
        print("vocabulary matches saved checkpoint")
    else:
        print("warning: vocabulary mismatch with saved checkpoint")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)'''
    nb.cells.append(new_code_cell(config_code))


def add_training_section(nb):
    """Add training loop."""
    
    training_code = '''# Training loop
if model_loaded:
    print("model already trained and loaded. skipping training.")
    print("to retrain, delete the checkpoint or change hyperparameters.")
    best_val_f1 = checkpoint.get("best_metric_value", 0.0)
    best_val_acc = checkpoint.get("best_metric_value", 0.0)
    print(f"best F1 from checkpoint: {best_val_f1:.4f}")
    print(f"best accuracy from checkpoint: {best_val_acc:.4f}")
else:
    print("starting training...")
    best_val_f1 = 0.0
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for x, y, mask, lengths, morf_feat in train_loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            lengths = lengths.to(device)
            morf_feat = morf_feat.to(device)

            logits = model(x, lengths, morf_feat)
            loss = masked_bce_loss(logits, y, mask)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()

        train_loss = total_loss / max(total_tokens, 1)

        model.eval()
        val_loss, val_tokens = 0.0, 0
        all_prec, all_rec, all_f1 = [], [], []
        with torch.no_grad():
            for x, y, mask, lengths, morf_feat in val_loader:
                x = x.to(device)
                y = y.to(device)
                mask = mask.to(device)
                lengths = lengths.to(device)
                morf_feat = morf_feat.to(device)

                logits = model(x, lengths, morf_feat)
                loss = masked_bce_loss(logits, y, mask)

                val_loss += loss.item() * mask.sum().item()
                val_tokens += mask.sum().item()

                p, r, f = boundary_f1(logits, y, mask, threshold=0.5)
                all_prec.append(p)
                all_rec.append(r)
                all_f1.append(f)

        val_loss = val_loss / max(val_tokens, 1)
        prec = np.mean(all_prec) if all_prec else 0.0
        rec = np.mean(all_rec) if all_rec else 0.0
        f1 = np.mean(all_f1) if all_f1 else 0.0

        acc = evaluate_accuracy_morf(acc_df, model, stoi, morf, device=device, threshold=0.5)

        print(f"epoch {epoch:02d} | train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"P={prec:.3f} R={rec:.3f} F1={f1:.3f}  Acc={acc:.3f}")

        if f1 > best_val_f1:
            best_val_f1 = f1
            save_model_checkpoint(
                model, stoi, itos, model_id,
                models_folder=MODELS_FOLDER,
                suffix="_best_f1",
                best_metric_value=best_val_f1
            )
            print("  ↳ saved checkpoint by F1 (best so far)")

        if acc > best_val_acc:
            best_val_acc = acc
            save_model_checkpoint(
                model, stoi, itos, model_id,
                models_folder=MODELS_FOLDER,
                suffix="_best_acc",
                best_metric_value=best_val_acc
            )
            print("  ↳ saved checkpoint by accuracy (best so far)")

    print(f"\\ntraining done!")
    print(f"best validation F1: {best_val_f1:.4f}")
    print(f"best validation accuracy: {best_val_acc:.4f}")'''
    nb.cells.append(new_code_cell(training_code))


def add_kfold_section(nb):
    """Add k-fold cross-validation function."""
    
    kfold_code = '''def run_kfold_cross_validation(
    df,
    n_folds=5,
    emb_dim=64,
    hidden_size=128,
    num_layers=2,
    dropout=0.3,
    freeze_emb=True,
    extra_feat_dim=1,
    epochs=20,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    morfessor_n_models=5,
    morfessor_seed_base=123,
    morfessor_lowercase=True,
    random_state=42,
    device=device
):
    """K-fold cross-validation with Morfessor ensemble features."""
    print(f"\\n{'=' * 80}")
    print(f"K-FOLD CV (k={n_folds}) WITH MORFESSOR ENSEMBLE")
    print(f"{'=' * 80}")
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    indices = np.arange(len(df))
    
    fold_results = []
    all_metrics = {
        'boundary_precision': [],
        'boundary_recall': [],
        'boundary_f1': [],
        'val_loss': []
    }
    
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(indices), 1):
        print(f"\\n--- fold {fold_idx}/{n_folds} ---")
        print(f"train: {len(train_indices)}, val: {len(val_indices)}")
        
        train_df_fold = df.iloc[train_indices].reset_index(drop=True)
        val_df_fold = df.iloc[val_indices].reset_index(drop=True)
        
        stoi_fold, itos_fold = build_vocab(train_df_fold["char_seq"].tolist())
        vocab_size = len(itos_fold)
        
        print(f"  training morfessor ensemble on fold {fold_idx}...")
        m_cfg_fold = MorfessorConfig(
            n_models=morfessor_n_models,
            seed_base=morfessor_seed_base + fold_idx,
            lowercase=morfessor_lowercase
        )
        morf_fold = MorfessorBoundaryFeaturizer(m_cfg_fold)
        morf_fold.fit(train_df_fold["Word"].tolist())
        print(f"  morfessor ensemble trained (n_models={morfessor_n_models})")
        
        pos_fold, neg_fold = count_pos_neg(train_df_fold)
        pos_weight_value_fold = float(neg_fold) / max(float(pos_fold), 1.0)
        
        def pad_batch_with_morf_fold(batch, pad_id=0):
            seqs, labels, morf = zip(*batch)
            x_ids = [[stoi_fold.get(t, stoi_fold["<UNK>"]) for t in s] for s in seqs]
            y_ids = [lab for lab in labels]
            m_probs = [mp for mp in morf]
            lengths = [len(x) for x in x_ids]
            maxlen = max(lengths)
            x_pad = [xi + [pad_id] * (maxlen - len(xi)) for xi in x_ids]
            y_pad = [yi + [0] * (maxlen - len(yi)) for yi in y_ids]
            m_pad = [mi + [0.0] * (maxlen - len(mi)) for mi in m_probs]
            mask = [[1] * len(xi) + [0] * (maxlen - len(xi)) for xi in x_ids]
            return (
                torch.LongTensor(x_pad),
                torch.FloatTensor(y_pad),
                torch.BoolTensor(mask),
                torch.LongTensor(lengths),
                torch.FloatTensor(m_pad).unsqueeze(-1),
            )
        
        train_ds_fold = CharBoundaryDatasetMorf(train_df_fold, morf_fold, stoi_fold)
        val_ds_fold = CharBoundaryDatasetMorf(val_df_fold, morf_fold, stoi_fold)
        train_loader_fold = DataLoader(
            train_ds_fold, batch_size=batch_size, shuffle=True,
            collate_fn=pad_batch_with_morf_fold
        )
        val_loader_fold = DataLoader(
            val_ds_fold, batch_size=batch_size, shuffle=False,
            collate_fn=pad_batch_with_morf_fold
        )
        
        model_fold = BiLSTMBoundaryWithMorf(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            freeze_emb=freeze_emb,
            extra_feat_dim=extra_feat_dim
        ).to(device)
        
        optimizer_fold = torch.optim.AdamW(model_fold.parameters(), lr=lr, weight_decay=weight_decay)
        
        def masked_bce_loss_fold(logits, targets, mask):
            loss_fn = nn.BCEWithLogitsLoss(
                reduction="none",
                pos_weight=torch.tensor(pos_weight_value_fold, device=logits.device)
            )
            loss_per_token = loss_fn(logits, targets) * mask.float()
            denom = mask.float().sum().clamp_min(1.0)
            return loss_per_token.sum() / denom
        
        best_val_f1 = 0.0
        best_val_prec = 0.0
        best_val_rec = 0.0
        best_val_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(1, epochs + 1):
            model_fold.train()
            total_loss = 0.0
            total_tokens = 0
            for x, y, mask, lengths, morf_feat in train_loader_fold:
                x = x.to(device)
                y = y.to(device)
                mask = mask.to(device)
                lengths = lengths.to(device)
                morf_feat = morf_feat.to(device)
                
                logits = model_fold(x, lengths, morf_feat)
                loss = masked_bce_loss_fold(logits, y, mask)
                
                optimizer_fold.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model_fold.parameters(), 1.0)
                optimizer_fold.step()
                
                total_loss += loss.item() * mask.sum().item()
                total_tokens += mask.sum().item()
            
            train_loss = total_loss / max(total_tokens, 1)
            
            model_fold.eval()
            val_loss, val_tokens = 0.0, 0
            all_prec, all_rec, all_f1 = [], [], []
            with torch.no_grad():
                for x, y, mask, lengths, morf_feat in val_loader_fold:
                    x = x.to(device)
                    y = y.to(device)
                    mask = mask.to(device)
                    lengths = lengths.to(device)
                    morf_feat = morf_feat.to(device)
                    
                    logits = model_fold(x, lengths, morf_feat)
                    loss = masked_bce_loss_fold(logits, y, mask)
                    val_loss += loss.item() * mask.sum().item()
                    val_tokens += mask.sum().item()
                    
                    p, r, f = boundary_f1(logits, y, mask, threshold=0.5)
                    all_prec.append(p)
                    all_rec.append(r)
                    all_f1.append(f)
            
            val_loss = val_loss / max(val_tokens, 1)
            prec = np.mean(all_prec) if all_prec else 0.0
            rec = np.mean(all_rec) if all_rec else 0.0
            f1 = np.mean(all_f1) if all_f1 else 0.0
            
            print(f"  ep {epoch:02d} | train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
            
            if f1 > best_val_f1 or (np.isclose(f1, best_val_f1) and val_loss < best_val_loss):
                best_val_f1 = f1
                best_val_prec = prec
                best_val_rec = rec
                best_val_loss = val_loss
                best_epoch = epoch
        
        print(f"\\n  best epoch: {best_epoch}")
        print(f"  best validation: P={best_val_prec:.3f} R={best_val_rec:.3f} F1={best_val_f1:.3f} Loss={best_val_loss:.4f}")
        
        fold_results.append({
            'fold': fold_idx,
            'boundary_precision': best_val_prec,
            'boundary_recall': best_val_rec,
            'boundary_f1': best_val_f1,
            'val_loss': best_val_loss,
            'best_epoch': best_epoch
        })
        
        all_metrics['boundary_precision'].append(best_val_prec)
        all_metrics['boundary_recall'].append(best_val_rec)
        all_metrics['boundary_f1'].append(best_val_f1)
        all_metrics['val_loss'].append(best_val_loss)
    
    mean_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    std_metrics = {k: np.std(v) for k, v in all_metrics.items()}
    best_fold_idx = max(range(len(fold_results)), key=lambda i: fold_results[i]['boundary_f1'])
    
    print(f"\\n{'=' * 80}")
    print("CV SUMMARY")
    print(f"{'=' * 80}")
    for r in fold_results:
        print(f"  fold {r['fold']}: P={r['boundary_precision']:.3f}, R={r['boundary_recall']:.3f}, "
              f"F1={r['boundary_f1']:.3f}, Loss={r['val_loss']:.4f}")
    
    print(f"\\nmean +/- std over {n_folds} folds:")
    print(f"  precision: {mean_metrics['boundary_precision']:.3f} +/- {std_metrics['boundary_precision']:.3f}")
    print(f"  recall:    {mean_metrics['boundary_recall']:.3f} +/- {std_metrics['boundary_recall']:.3f}")
    print(f"  F1:        {mean_metrics['boundary_f1']:.3f} +/- {std_metrics['boundary_f1']:.3f}")
    print(f"  loss:      {mean_metrics['val_loss']:.4f} +/- {std_metrics['val_loss']:.4f}")
    print(f"\\nbest fold: {fold_results[best_fold_idx]['fold']} (F1={fold_results[best_fold_idx]['boundary_f1']:.3f})")
    print(f"{'=' * 80}\\n")
    
    return {
        'fold_results': fold_results,
        'mean_metrics': mean_metrics,
        'std_metrics': std_metrics,
        'best_fold_idx': best_fold_idx,
        'all_metrics': all_metrics
    }'''
    nb.cells.append(new_code_cell(kfold_code))


def add_run_kfold_section(nb):
    """Add k-fold CV execution."""
    
    run_kfold = '''# Run k-fold cross-validation
kfold_results = run_kfold_cross_validation(
    df=gold_df,
    n_folds=5,
    emb_dim=EMB_DIM,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    freeze_emb=FREEZE_EMB,
    extra_feat_dim=EXTRA_FEAT_DIM,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    morfessor_n_models=m_cfg.n_models,
    morfessor_seed_base=m_cfg.seed_base,
    morfessor_lowercase=m_cfg.lowercase,
    random_state=42,
    device=device
)

print(f"\\navg boundary F1: {kfold_results['mean_metrics']['boundary_f1']:.3f} +/- {kfold_results['std_metrics']['boundary_f1']:.3f}")
print(f"avg precision: {kfold_results['mean_metrics']['boundary_precision']:.3f} +/- {kfold_results['std_metrics']['boundary_precision']:.3f}")
print(f"avg recall: {kfold_results['mean_metrics']['boundary_recall']:.3f} +/- {kfold_results['std_metrics']['boundary_recall']:.3f}")'''
    nb.cells.append(new_code_cell(run_kfold))


def add_example_usage_section(nb):
    """Add example word segmentation."""
    
    example = '''# Example predictions
test_words = ["rikuchkani", "pikunas", "ñichkanchus"]
pred_b = predict_boundaries_with_morf(test_words, model, stoi, morf, threshold=0.5)
for w, b in zip(test_words, pred_b):
    toks = to_graphemes_quechua(w)
    print(f"{w} {b} -> {apply_boundaries_tokens(toks, b)}")'''
    nb.cells.append(new_code_cell(example))


def add_test_loading_section(nb):
    """Add test data loading."""
    
    test_load = '''# Load test data
print("loading test data...")
acc_df = pd.read_parquet(os.path.join(DATA_FOLDER, "cleaned_data_df.parquet"))
print(f"loaded {len(acc_df):,} test examples")'''
    nb.cells.append(new_code_cell(test_load))


def add_evaluation_section(nb):
    """Add evaluation functions and test set evaluation."""
    
    eval_helpers = '''def boundary_positions_from_labels(labels, L=None):
    """Convert per-token boundary labels to boundary positions."""
    if not labels:
        return set()
    if L is None:
        L = len(labels)
    upto = min(L - 1, len(labels))
    return {i for i in range(upto) if labels[i] == 1}

def boundary_positions_from_morpheme_tokens(morpheme_token_lists):
    """Given morpheme token lists, return boundary positions."""
    pos = set()
    acc = 0
    for k, toks in enumerate(morpheme_token_lists):
        acc += len(toks)
        if k < len(morpheme_token_lists) - 1:
            pos.add(acc - 1)
    return pos

def prf_from_sets(pred_set, gold_set):
    """Compute precision, recall, F1 from boundary sets."""
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    if tp + fp == 0:
        precision = 1.0 if (tp + fn == 0) else 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 1.0 if (tp + fp == 0) else 0.0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        f1 = 1.0 if (tp + fp + fn) == 0 else 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return tp, fp, fn, precision, recall, f1

def best_variant_metrics_token_space(word_tokens, pred_boundary_labels, gold_variants):
    """Compare predicted boundaries to gold variants, pick best F1."""
    pred_b = boundary_positions_from_labels(pred_boundary_labels, L=len(word_tokens))

    best = None
    for variant in gold_variants:
        variant_token_lists = [to_graphemes_quechua(m) for m in variant]
        gold_b = boundary_positions_from_morpheme_tokens(variant_token_lists)
        tp, fp, fn, P, R, F1 = prf_from_sets(pred_b, gold_b)
        key = (F1, tp, -fn, -fp)
        if (best is None) or (key > best[0]):
            best = (key, gold_b, tp, fp, fn, P, R, F1)

    if best is None:
        gold_b = set()
        tp, fp, fn, P, R, F1 = prf_from_sets(pred_b, gold_b)
        return pred_b, gold_b, tp, fp, fn, P, R, F1

    _, gold_b, tp, fp, fn, P, R, F1 = best
    return pred_b, gold_b, tp, fp, fn, P, R, F1

def is_correct_prediction(predicted, gold_variants):
    """Check if predicted segmentation matches any gold variant."""
    return any(predicted == variant for variant in gold_variants)

def split_count_metrics(predicted_segments, gold_variants):
    """Compute split-count accuracy variants."""
    pred_count = len(predicted_segments)
    gold_counts = [len(gold) for gold in gold_variants]

    exact = any(pred_count == g for g in gold_counts)
    plus1 = any(pred_count == g + 1 for g in gold_counts)
    minus1 = any(pred_count == g - 1 for g in gold_counts)
    plusminus1 = any(abs(pred_count - g) <= 1 for g in gold_counts)

    return {"Exact": exact, "+1": plus1, "-1": minus1, "±1": plusminus1}'''
    nb.cells.append(new_code_cell(eval_helpers))
    
    test_eval = '''# Load trained model checkpoint
print(f"loading model {model_id}...")
ckpt = load_model_checkpoint(model_id, models_folder=MODELS_FOLDER, suffix="_best_acc")

if ckpt is None:
    raise FileNotFoundError(f"model checkpoint not found for model_id: {model_id}")

stoi, itos = ckpt["stoi"], ckpt["itos"]
model.load_state_dict(ckpt["model_state"])
model.eval()
print("model loaded successfully!")

# Evaluate on test set
all_words = acc_df["Word"].tolist()
all_gold = acc_df["Gold"]

all_boundaries = predict_boundaries_with_morf(
    all_words, model, stoi, morf, threshold=0.5, device=device
)

records = []
micro_tp = micro_fp = micro_fn = 0
macro_Ps, macro_Rs, macro_F1s = [], [], []
exact_flags = []
split_exact_flags = []
split_plus1_flags = []
split_minus1_flags = []
split_pm1_flags = []
overlap_flags = []

for word, gold_variants, boundary_labels in zip(all_words, all_gold, all_boundaries):
    gold_variants = normalize_gold_variants(gold_variants)
    
    toks = to_graphemes_quechua(word)
    predicted_segments = apply_boundaries_tokens(toks, boundary_labels)

    correct_exact = is_correct_prediction(predicted_segments, gold_variants)
    split_metrics = split_count_metrics(predicted_segments, gold_variants)
    overlap = correct_exact and split_metrics["Exact"]

    pred_b, gold_b_chosen, tp, fp, fn, P, R, F1 = best_variant_metrics_token_space(
        toks, boundary_labels, gold_variants
    )

    records.append({
        "Word": word,
        "Prediction": predicted_segments,
        "Gold": gold_variants,
        "PredBoundaries(tok_idx)": sorted(pred_b),
        "GoldBoundaries(Chosen tok_idx)": sorted(gold_b_chosen),
        "TP": tp, "FP": fp, "FN": fn,
        "P_word": P, "R_word": R, "F1_word": F1,
        "CorrectExactSeg": correct_exact,
        "CorrectSplitCount": split_metrics["Exact"],
        "SplitCount+1": split_metrics["+1"],
        "SplitCount-1": split_metrics["-1"],
        "SplitCount±1": split_metrics["±1"],
        "OverlapExactAndSplit": overlap
    })

    micro_tp += tp
    micro_fp += fp
    micro_fn += fn
    macro_Ps.append(P)
    macro_Rs.append(R)
    macro_F1s.append(F1)
    exact_flags.append(correct_exact)
    split_exact_flags.append(split_metrics["Exact"])
    split_plus1_flags.append(split_metrics["+1"])
    split_minus1_flags.append(split_metrics["-1"])
    split_pm1_flags.append(split_metrics["±1"])
    overlap_flags.append(overlap)

results_df = pd.DataFrame(records)

accuracy = results_df["CorrectExactSeg"].mean()

if micro_tp + micro_fp == 0:
    P_micro = 1.0 if micro_tp + micro_fn == 0 else 0.0
else:
    P_micro = micro_tp / (micro_tp + micro_fp)

if micro_tp + micro_fn == 0:
    R_micro = 1.0 if micro_tp + micro_fp == 0 else 0.0
else:
    R_micro = micro_tp / (micro_tp + micro_fn)

if P_micro + R_micro == 0:
    F1_micro = 1.0 if (micro_tp + micro_fp + micro_fn) == 0 else 0.0
else:
    F1_micro = 2 * P_micro * R_micro / (P_micro + R_micro)

P_macro = float(pd.Series(macro_Ps).mean()) if macro_Ps else 0.0
R_macro = float(pd.Series(macro_Rs).mean()) if macro_Rs else 0.0
F1_macro = float(pd.Series(macro_F1s).mean()) if macro_F1s else 0.0

exact_accuracy = np.mean(exact_flags)
split_exact_acc = np.mean(split_exact_flags)
split_plus1_acc = np.mean(split_plus1_flags)
split_minus1_acc = np.mean(split_minus1_flags)
split_pm1_acc = np.mean(split_pm1_flags)
overlap_accuracy = np.mean(overlap_flags)

print("=== segmentation and split count metrics ===")
print(f"exact segmentation accuracy:  {exact_accuracy:.4f}")
print(f"split-count (exact):          {split_exact_acc:.4f}")
print(f"split-count (+1):             {split_plus1_acc:.4f}")
print(f"split-count (−1):             {split_minus1_acc:.4f}")
print(f"split-count (±1):             {split_pm1_acc:.4f}")
print(f"overlap (exact ∩ split):      {overlap_accuracy:.4f}")

print("boundary metrics (token space):")
print(f"  micro  - P: {P_micro:.4f}  R: {R_micro:.4f}  F1: {F1_micro:.4f}")
print(f"  macro  - P: {P_macro:.4f}  R: {R_macro:.4f}  F1: {F1_macro:.4f}")

# Save results
results_output_path = os.path.join(DATA_FOLDER, "bilstm_morfessor_eval_results.csv")
results_df.to_csv(results_output_path, index=False)
print(f"\\nevaluation results saved to {results_output_path}")'''
    nb.cells.append(new_code_cell(test_eval))


def main():
    """Build and save the refactored notebook."""
    nb = create_notebook()
    
    # Add sections in order
    add_header_section(nb)
    add_imports_section(nb)
    add_config_section(nb)
    add_text_normalization_section(nb)
    add_data_loading_section(nb)
    add_morfessor_section(nb)
    add_vocabulary_section(nb)
    add_morfessor_io_section(nb)
    add_train_morfessor_section(nb)
    add_dataset_section(nb)
    add_model_section(nb)
    add_loss_section(nb)
    add_metrics_section(nb)
    add_inference_section(nb)
    add_checkpointing_section(nb)
    add_model_config_section(nb)
    add_training_section(nb)
    add_kfold_section(nb)
    add_run_kfold_section(nb)
    add_example_usage_section(nb)
    add_test_loading_section(nb)
    add_evaluation_section(nb)
    
    # Save notebook
    output_path = "segmenter-morfessor_refactored.ipynb"
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print(f"Refactored notebook saved to: {output_path}")


if __name__ == "__main__":
    main()

