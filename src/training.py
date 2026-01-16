"""
Training utilities for Quechua morphological segmentation models.

Includes:
- PyTorch Dataset classes
- Training loops
- Checkpoint management
- Threshold tuning
"""

import os
import json
import hashlib
import pickle
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

from .preprocessing import (
    to_graphemes, get_boundary_labels, build_vocab,
    encode_sequence, apply_boundaries, PAD, UNK
)
from .evaluation import evaluate_predictions, compute_boundary_prf


# ==============================================================================
# Datasets
# ==============================================================================

class BoundaryDataset(Dataset):
    """
    PyTorch Dataset for boundary prediction.
    
    Each sample is a (token_sequence, boundary_labels) pair.
    """
    
    def __init__(
        self,
        words: List[str],
        morph_splits: List[List[str]],
        tokenize_fn=None
    ):
        """
        Args:
            words: List of surface forms
            morph_splits: List of morpheme lists
            tokenize_fn: Tokenization function (default: to_graphemes)
        """
        self.tokenize_fn = tokenize_fn or to_graphemes
        
        self.token_seqs = []
        self.labels = []
        
        for word, morphs in zip(words, morph_splits):
            tokens = self.tokenize_fn(word)
            morph_tokens = [self.tokenize_fn(m) for m in morphs]
            boundary = get_boundary_labels(tokens, morph_tokens)
            
            self.token_seqs.append(tokens)
            self.labels.append(boundary)
    
    def __len__(self):
        return len(self.token_seqs)
    
    def __getitem__(self, idx):
        return self.token_seqs[idx], self.labels[idx]


class Seq2SeqDataset(Dataset):
    """
    PyTorch Dataset for seq2seq segmentation.
    
    Input: character sequence
    Output: segmented string with '+' separators
    """
    
    def __init__(
        self,
        words: List[str],
        morph_splits: List[List[str]],
        input_vocab: Dict[str, int] = None,
        output_vocab: Dict[str, int] = None
    ):
        self.words = words
        self.targets = ["+".join(m) for m in morph_splits]
        
        # Build vocabularies if not provided
        if input_vocab is None:
            all_chars = set()
            for w in words:
                all_chars.update(list(w.lower()))
            self.input_vocab = {PAD: 0, UNK: 1, "<SOS>": 2, "<EOS>": 3}
            for c in sorted(all_chars):
                self.input_vocab[c] = len(self.input_vocab)
        else:
            self.input_vocab = input_vocab
        
        if output_vocab is None:
            all_chars = set()
            for t in self.targets:
                all_chars.update(list(t.lower()))
            self.output_vocab = {PAD: 0, UNK: 1, "<SOS>": 2, "<EOS>": 3}
            for c in sorted(all_chars):
                self.output_vocab[c] = len(self.output_vocab)
        else:
            self.output_vocab = output_vocab
    
    def __len__(self):
        return len(self.words)
    
    def __getitem__(self, idx):
        src = [self.input_vocab.get(c, 1) for c in self.words[idx].lower()]
        tgt = [self.output_vocab["<SOS>"]] + \
              [self.output_vocab.get(c, 1) for c in self.targets[idx].lower()] + \
              [self.output_vocab["<EOS>"]]
        return torch.tensor(src), torch.tensor(tgt)


# ==============================================================================
# Collate Functions
# ==============================================================================

def boundary_collate_fn(batch, stoi: Dict[str, int], pad_id: int = 0):
    """
    Collate function for boundary prediction.
    
    Pads sequences and creates attention masks.
    """
    seqs, labels = zip(*batch)
    
    x_ids = [encode_sequence(s, stoi) for s in seqs]
    lengths = [len(x) for x in x_ids]
    max_len = max(lengths)
    
    x_pad = [xi + [pad_id] * (max_len - len(xi)) for xi in x_ids]
    y_pad = [yi + [0] * (max_len - len(yi)) for yi in labels]
    mask = [[1] * len(xi) + [0] * (max_len - len(xi)) for xi in x_ids]
    
    return (
        torch.LongTensor(x_pad),
        torch.FloatTensor(y_pad),
        torch.BoolTensor(mask),
        torch.LongTensor(lengths)
    )


def seq2seq_collate_fn(batch, pad_id: int = 0):
    """Collate function for seq2seq models."""
    srcs, tgts = zip(*batch)
    
    src_lens = [len(s) for s in srcs]
    tgt_lens = [len(t) for t in tgts]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)
    
    src_pad = torch.zeros(len(batch), max_src, dtype=torch.long)
    tgt_pad = torch.zeros(len(batch), max_tgt, dtype=torch.long)
    
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_pad[i, :len(s)] = s
        tgt_pad[i, :len(t)] = t
    
    src_mask = (src_pad == pad_id)
    tgt_mask = (tgt_pad == pad_id)
    
    return src_pad, tgt_pad, src_mask, tgt_mask


# ==============================================================================
# Training Functions
# ==============================================================================

def train_boundary_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    pos_weight: float = 3.0,
    device: str = "cpu",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train a boundary prediction model.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization
        pos_weight: Weight for positive (boundary) class
        device: Training device
        verbose: Print progress
        
    Returns:
        Training history dict
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
        "val_exact": [],
        "best_epoch": 0,
        "best_val_f1": 0
    }
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        
        for x, y, mask, lengths in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            lengths = lengths.to(device)
            
            optimizer.zero_grad()
            logits = model(x, lengths)
            
            # Masked loss
            loss = criterion(logits[mask], y[mask])
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_tp, all_fp, all_fn = 0, 0, 0
        
        with torch.no_grad():
            for x, y, mask, lengths in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                lengths = lengths.to(device)
                
                logits = model(x, lengths)
                loss = criterion(logits[mask], y[mask])
                val_loss += loss.item()
                
                # Compute boundary metrics
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                for i in range(len(x)):
                    length = lengths[i].item()
                    pred_b = set(j for j in range(length - 1) if preds[i, j] > 0.5)
                    gold_b = set(j for j in range(length - 1) if y[i, j] > 0.5)
                    
                    tp = len(pred_b & gold_b)
                    fp = len(pred_b - gold_b)
                    fn = len(gold_b - pred_b)
                    
                    all_tp += tp
                    all_fp += fp
                    all_fn += fn
        
        val_loss /= len(val_loader)
        
        # Compute F1
        precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
        recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        history["val_loss"].append(val_loss)
        history["val_f1"].append(f1)
        
        if f1 > history["best_val_f1"]:
            history["best_val_f1"] = f1
            history["best_epoch"] = epoch
        
        if verbose:
            print(f"  ep {epoch:02d} | loss={train_loss:.4f} | P/R/F1={precision:.3f}/{recall:.3f}/{f1:.3f}")
    
    return history


def tune_threshold(
    model: nn.Module,
    val_loader: DataLoader,
    stoi: Dict[str, int],
    device: str = "cpu",
    thresholds: List[float] = None
) -> Tuple[float, float]:
    """
    Tune decision threshold on validation set.
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        stoi: Token-to-index mapping
        device: Device
        thresholds: List of thresholds to try
        
    Returns:
        Tuple of (best_threshold, best_f1)
    """
    if thresholds is None:
        thresholds = np.arange(0.3, 0.7, 0.02).tolist()
    
    model.eval()
    
    # Collect all predictions
    all_logits = []
    all_labels = []
    all_lengths = []
    
    with torch.no_grad():
        for x, y, mask, lengths in val_loader:
            x = x.to(device)
            lengths = lengths.to(device)
            
            logits = model(x, lengths)
            
            all_logits.append(logits.cpu())
            all_labels.append(y)
            all_lengths.append(lengths.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_lengths = torch.cat(all_lengths, dim=0)
    
    probs = torch.sigmoid(all_logits)
    
    best_thr = 0.5
    best_f1 = 0
    
    for thr in thresholds:
        tp, fp, fn = 0, 0, 0
        
        for i in range(len(probs)):
            length = all_lengths[i].item()
            pred_b = set(j for j in range(length - 1) if probs[i, j] > thr)
            gold_b = set(j for j in range(length - 1) if all_labels[i, j] > 0.5)
            
            tp += len(pred_b & gold_b)
            fp += len(pred_b - gold_b)
            fn += len(gold_b - pred_b)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    
    return best_thr, best_f1


# ==============================================================================
# Checkpoint Management
# ==============================================================================

def generate_model_id(**params) -> str:
    """Generate unique model ID from hyperparameters."""
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()[:16]


def save_checkpoint(
    model: nn.Module,
    stoi: Dict[str, int],
    itos: List[str],
    model_id: str,
    save_dir: str,
    extra_data: Dict = None
):
    """Save model checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, model_id)
    os.makedirs(model_path, exist_ok=True)
    
    # Save model state
    torch.save(model.state_dict(), os.path.join(model_path, "model.pt"))
    
    # Save vocabulary
    vocab_data = {"stoi": stoi, "itos": itos}
    if extra_data:
        vocab_data.update(extra_data)
    
    with open(os.path.join(model_path, "vocab.json"), "w") as f:
        json.dump(vocab_data, f)
    
    print(f"Saved checkpoint to {model_path}")


def load_checkpoint(
    model_id: str,
    save_dir: str
) -> Optional[Dict]:
    """Load model checkpoint."""
    model_path = os.path.join(save_dir, model_id)
    
    if not os.path.exists(model_path):
        return None
    
    # Load model state
    model_state = torch.load(os.path.join(model_path, "model.pt"), map_location="cpu")
    
    # Load vocabulary
    with open(os.path.join(model_path, "vocab.json"), "r") as f:
        vocab_data = json.load(f)
    
    return {
        "model_state": model_state,
        "stoi": vocab_data["stoi"],
        "itos": vocab_data["itos"],
        **{k: v for k, v in vocab_data.items() if k not in ["stoi", "itos"]}
    }


# ==============================================================================
# Cross-Validation
# ==============================================================================

def run_kfold_cv(
    model_class,
    model_kwargs: Dict,
    words: List[str],
    morph_splits: List[List[str]],
    n_folds: int = 5,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    random_state: int = 42,
    verbose: bool = True
) -> List[Dict]:
    """
    Run k-fold cross-validation.
    
    Args:
        model_class: Model class to instantiate
        model_kwargs: Keyword arguments for model
        words: List of words
        morph_splits: List of morpheme lists
        n_folds: Number of CV folds
        epochs: Training epochs per fold
        batch_size: Batch size
        lr: Learning rate
        device: Training device
        random_state: Random seed
        verbose: Print progress
        
    Returns:
        List of fold result dicts
    """
    from functools import partial
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    indices = np.arange(len(words))
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(indices), 1):
        if verbose:
            print(f"\n--- Fold {fold_idx}/{n_folds} ---")
        
        # Split data
        train_words = [words[i] for i in train_idx]
        train_morphs = [morph_splits[i] for i in train_idx]
        val_words = [words[i] for i in val_idx]
        val_morphs = [morph_splits[i] for i in val_idx]
        
        # Create datasets
        train_ds = BoundaryDataset(train_words, train_morphs)
        val_ds = BoundaryDataset(val_words, val_morphs)
        
        # Build vocabulary from training
        stoi, itos = build_vocab(train_ds.token_seqs)
        
        # Create data loaders
        collate = partial(boundary_collate_fn, stoi=stoi)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate)
        
        # Create model
        model = model_class(vocab_size=len(itos), **model_kwargs)
        
        # Train
        history = train_boundary_model(
            model, train_loader, val_loader,
            epochs=epochs, lr=lr, device=device, verbose=verbose
        )
        
        fold_results.append({
            "fold": fold_idx,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "best_epoch": history["best_epoch"],
            "best_val_f1": history["best_val_f1"],
            "history": history
        })
    
    # Summary
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"CV Summary")
        print(f"{'=' * 60}")
        f1s = [r["best_val_f1"] for r in fold_results]
        print(f"Mean F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"{'=' * 60}")
    
    return fold_results
