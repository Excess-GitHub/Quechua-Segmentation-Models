#!/usr/bin/env python3
"""
Script to generate a refactored version of the Markov-LSTM-MarkovFilter notebook.
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
        "# Markov-LSTM-Markov Filter: Quechua Morphology Parser\n\n"
        "Morphological segmentation for Quechua using:\n"
        "- BiLSTM for boundary prediction\n"
        "- HMM priors from suffix patterns\n"
        "- K-teacher regularization"
    ))


def add_imports_section(nb):
    """Add all imports in a single organized cell."""
    imports_code = '''# Core libraries
import os
import re
import ast
import json
import math
import hashlib
import pickle
from collections import Counter, defaultdict
from typing import List, Set, Tuple

# Data handling
import numpy as np
import pandas as pd

# ML & DL
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, GroupShuffleSplit
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support'''
    nb.cells.append(new_code_cell(imports_code))


def add_config_section(nb):
    """Add configuration constants."""
    config_code = '''# Paths
DATA_FOLDER = "data"
MODEL_NAME = "Markov-LSTM-MarkovFilter"
MODELS_FOLDER = f"models_{MODEL_NAME}"
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Synthetic data options: "none", "gpt4o", "gpt5mini"
SYNTHETIC_DATA_CHOICE = "none"

# Word selection for augmentation: "all", "first", "random"
AUGMENTATION_WORD_SELECTION = "random"
AUGMENTATION_N_WORDS = 100

# Random seed
RNG = 42
torch.manual_seed(RNG)
np.random.seed(RNG)

# Feature columns used for privileged knowledge
NEW_NUM_FEATS = [
    "Word_len", "Vowel_no", "Cons_no",
    "Tail_cons_no", "Tail_vowel_no",
    "No_splits", "YW_count", "Tail_YW_count"
]

# Quechua graphemes for tokenization
graphemes = [
    "ch", "ll", "rr", "tr", "kw", "ph",
    "a", "b", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "ñ", "o", "p", "q",
    "r", "s", "t", "u", "v", "w", "x", "y"
]'''
    nb.cells.append(new_code_cell(config_code))


def add_data_loading_section(nb):
    """Add data loading functions and cells."""
    
    # Gold data loading
    gold_loading = '''# Load the gold standard segmentations
print("loading gold data...")
gold_df = pd.read_parquet(os.path.join(DATA_FOLDER, "Sue_kalt.parquet"))
gold_df['Word'] = gold_df['word']
gold_df['morph'] = gold_df['morph'].str.replace('-', ' ')
gold_df['Morph_split_str'] = gold_df['morph']
gold_df['Morph_split'] = gold_df['morph'].str.split(' ')
gold_df = gold_df[['Word', 'Morph_split', 'Morph_split_str']]
gold_df.drop_duplicates(subset='Word', keep='first', inplace=True)
gold_df.dropna(subset=['Word'], inplace=True)
print(f"got {len(gold_df):,} gold examples")'''
    nb.cells.append(new_code_cell(gold_loading))
    
    # Synthetic data loader function
    synthetic_loader = '''def load_synthetic_data(choice):
    """Load GPT-generated segmentations if augmentation is enabled."""
    if choice == "none":
        print("no synthetic augmentation")
        return None
    
    file_map = {
        "gpt4o": "gpt4o_synthetic_segmentations.csv",
        "gpt5mini": "gpt5mini_synthetic_segmentations.csv"
    }
    
    if choice not in file_map:
        print(f"unknown choice '{choice}', skipping augmentation")
        return None
    
    file_path = os.path.join(DATA_FOLDER, file_map[choice])
    if not os.path.exists(file_path):
        print(f"file not found: {file_path}")
        return None
    
    print(f"loading synthetic data from {file_path}...")
    df = pd.read_csv(file_path)
    df = df.drop_duplicates(subset=['Original_Word']).reset_index(drop=True)
    
    # Filter out garbage responses
    bad_strings = ['can\\'t', 'quechua', 'sorry', 'could']
    df = df[~df['Segmented_Morphemes'].str.contains('|'.join(bad_strings), case=False, na=False)]
    
    df = df.rename(columns={'Original_Word': 'Word'})
    df['Morph_split_str'] = df['Segmented_Morphemes']
    df['Morph_split'] = df['Segmented_Morphemes'].str.split(' ')
    df = df[['Word', 'Morph_split', 'Morph_split_str']]
    
    print(f"loaded {len(df):,} synthetic examples")
    return df

synthetic_df = load_synthetic_data(SYNTHETIC_DATA_CHOICE)'''
    nb.cells.append(new_code_cell(synthetic_loader))
    
    # GPT data for comparison
    gpt_data = '''# Load GPT data separately for finding common words
gpt_5_mini_df = pd.read_csv(os.path.join(DATA_FOLDER, "gpt5mini_synthetic_segmentations.csv"))
gpt_5_mini_df = gpt_5_mini_df.drop_duplicates(subset=['Original_Word']).reset_index(drop=True)
bad_strings = ['can\\'t', 'quechua', 'sorry', 'could']
gpt_5_mini_df = gpt_5_mini_df[~gpt_5_mini_df['Segmented_Morphemes'].str.contains('|'.join(bad_strings), case=False, na=False)]
gpt_5_mini_df = gpt_5_mini_df.rename(columns={'Original_Word': 'Word'})
gpt_5_mini_df['Morph_split_str'] = gpt_5_mini_df['Segmented_Morphemes']
gpt_5_mini_df['Morph_split'] = gpt_5_mini_df['Segmented_Morphemes'].str.split(' ')
gpt_5_mini_df = gpt_5_mini_df[['Word', 'Morph_split', 'Morph_split_str']]

gpt_4o_df = pd.read_csv(os.path.join(DATA_FOLDER, "gpt4o_synthetic_segmentations.csv"))
gpt_4o_df = gpt_4o_df.drop_duplicates(subset=['Original_Word']).reset_index(drop=True)
gpt_4o_df = gpt_4o_df[~gpt_4o_df['Segmented_Morphemes'].str.contains('|'.join(bad_strings), case=False, na=False)]
gpt_4o_df = gpt_4o_df.rename(columns={'Original_Word': 'Word'})
gpt_4o_df['Morph_split_str'] = gpt_4o_df['Segmented_Morphemes']
gpt_4o_df['Morph_split'] = gpt_4o_df['Segmented_Morphemes'].str.split(' ')
gpt_4o_df = gpt_4o_df[['Word', 'Morph_split', 'Morph_split_str']]

gpt_5_mini_words = set(gpt_5_mini_df['Word'])
gpt_4o_words = set(gpt_4o_df['Word'])
common_words = gpt_4o_words.intersection(gpt_5_mini_words)
print(f"words in both GPT sets: {len(common_words)}")'''
    nb.cells.append(new_code_cell(gpt_data))
    
    # Combine data
    combine_data = '''# Merge synthetic with gold if augmentation is on
if synthetic_df is not None:
    gpt_5_mini_words = set(gpt_5_mini_df['Word'])
    gpt_4o_words = set(gpt_4o_df['Word'])
    common_words = gpt_4o_words.intersection(gpt_5_mini_words)
    print(f"common words between GPT models: {len(common_words):,}")
    
    if AUGMENTATION_WORD_SELECTION == "all":
        selected_words = common_words
        print(f"using all {len(selected_words):,} common words")
    elif AUGMENTATION_WORD_SELECTION == "first":
        sorted_words = sorted(common_words)
        n = min(AUGMENTATION_N_WORDS, len(sorted_words))
        selected_words = set(sorted_words[:n])
        print(f"using first {n:,} words alphabetically")
    elif AUGMENTATION_WORD_SELECTION == "random":
        import random
        seed = RNG if 'RNG' in globals() else 42
        random.seed(seed)
        n = min(AUGMENTATION_N_WORDS, len(common_words))
        selected_words = set(random.sample(list(common_words), n))
        print(f"using {n:,} random words")
    else:
        print(f"unknown selection '{AUGMENTATION_WORD_SELECTION}', using all")
        selected_words = common_words
    
    if SYNTHETIC_DATA_CHOICE in ["gpt5mini", "gpt4o"]:
        df_sampled = synthetic_df[synthetic_df['Word'].isin(selected_words)]
    else:
        df_sampled = None
    
    if df_sampled is not None and len(df_sampled) > 0:
        gold_df = pd.concat([df_sampled, gold_df], ignore_index=True)
        print(f"combined: {len(gold_df):,} total examples")
    else:
        print("no synthetic data added")
else:
    print("gold data only (no augmentation)")'''
    nb.cells.append(new_code_cell(combine_data))
    
    # Save common words
    save_common = '''# Save sampled words for reference
if synthetic_df is not None and 'df_sampled' in locals() and df_sampled is not None:
    df_sampled = df_sampled.sort_values(by="Word")
    output_file = os.path.join(DATA_FOLDER, f"{SYNTHETIC_DATA_CHOICE}_common.parquet")
    df_sampled.to_parquet(output_file, index=False)
    print(f"saved common words to {output_file}")'''
    nb.cells.append(new_code_cell(save_common))
    
    # Test data
    test_data = '''# Load test set
acc_df = pd.read_parquet(os.path.join(DATA_FOLDER, "cleaned_data_df.parquet"))

print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)
print(f"training: {gold_df.shape}")
print(f"test: {acc_df.shape}")
print(f"augmentation: {SYNTHETIC_DATA_CHOICE}")
print("=" * 60)'''
    nb.cells.append(new_code_cell(test_data))


def add_preprocessing_section(nb):
    """Add tokenization and feature extraction."""
    
    tokenize_code = '''pattern = re.compile("|".join(sorted(graphemes, key=len, reverse=True)))

def tokenize_morphemes(morphs):
    """Break morphemes into grapheme tokens."""
    return [pattern.findall(m.lower()) for m in morphs]

gold_df["Char_split"] = gold_df["Morph_split"].apply(tokenize_morphemes)'''
    nb.cells.append(new_code_cell(tokenize_code))
    
    cv_code = '''vowels = {"a", "i", "e", "o", "u"}

def grapheme_to_cv(grapheme):
    return "V" if grapheme in vowels else "C"

def morphs_to_cv(morphs):
    """Convert grapheme lists to CV (consonant/vowel) patterns."""
    return [[grapheme_to_cv(g) for g in morph] for morph in morphs]

gold_df["CV_split"] = gold_df["Char_split"].apply(morphs_to_cv)

def cv_to_string(cv_split):
    """Turn nested CV list into dash-separated string."""
    return "-".join("".join(m) for m in cv_split)'''
    nb.cells.append(new_code_cell(cv_code))
    
    features_code = '''# Build the feature dataframe
str_df = pd.DataFrame()
str_df["Full_chain"] = gold_df["CV_split"].apply(cv_to_string)
str_df["Trimmed_chain"] = str_df["Full_chain"].apply(
    lambda x: x.split("-", 1)[1] if "-" in x else np.nan
)
str_df["Word"] = gold_df["Word"]
str_df["Char_split"] = gold_df["Char_split"]
str_df["Morph_split"] = gold_df["Morph_split"]
str_df = str_df.dropna(subset=["Trimmed_chain"]).reset_index(drop=True)

# Numeric features
str_df["Word_len"] = str_df["Word"].str.len()
str_df["Vowel_no"] = str_df["Full_chain"].str.count("V")
str_df["Cons_no"] = str_df["Full_chain"].str.count("C")
str_df["Tail_cons_no"] = str_df["Trimmed_chain"].str.count("C")
str_df["Tail_vowel_no"] = str_df["Trimmed_chain"].str.count("V")
str_df["No_splits"] = str_df["Morph_split"].str.len()
str_df["YW_count"] = str_df["Word"].str.count("[yw]")
str_df["Tail_YW_count"] = str_df["Morph_split"].apply(
    lambda ms: sum(m.count("y") + m.count("w") for m in ms[1:])
)

str_df.head()'''
    nb.cells.append(new_code_cell(features_code))


def add_helper_functions_section(nb):
    """Add utility functions."""
    
    helpers = '''def safe_list(x):
    """Handle various list-like formats from dataframes."""
    if isinstance(x, list):
        return x
    s = str(x)
    try:
        return ast.literal_eval(s)
    except Exception:
        s2 = s.replace("[[", "[['").replace("]]", "']]").replace("], [", "'],['").replace(", ", "','")
        return ast.literal_eval(s2)

def flatten(list_of_lists):
    """Flatten nested list into single list of strings."""
    out = []
    for seg in list_of_lists:
        out.extend(seg)
    return [str(t) for t in out]

def extract_priv_features_from_row(row, feat_names):
    """Pull numeric features from a row into a vector."""
    vec = []
    for k in feat_names:
        val = row[k] if (k in row and pd.notna(row[k])) else 0.0
        try:
            vec.append(float(val))
        except Exception:
            vec.append(0.0)
    return vec'''
    nb.cells.append(new_code_cell(helpers))


def add_hmm_prior_section(nb):
    """Add HMM prior model code."""
    
    hmm_class = '''class SuffixHMMPrior:
    """
    Forward-backward algorithm over a suffix vocabulary.
    Gives boundary probabilities based on how well positions
    align with known suffix patterns.
    """
    def __init__(self, suffix_log_probs, max_suffix_len, unk_penalty=-15.0):
        self.log_probs = suffix_log_probs
        self.max_len = max_suffix_len
        self.unk_penalty = unk_penalty
        self.LOG_ZERO = -1e9

    def _get_log_prob(self, segment):
        return self.log_probs.get(segment, self.unk_penalty)

    def _forward_pass(self, word):
        n = len(word)
        alpha = [self.LOG_ZERO] * (n + 1)
        alpha[0] = 0.0

        for i in range(1, n + 1):
            log_sums = []
            for j in range(max(0, i - self.max_len), i):
                segment = word[j:i]
                log_p_segment = self._get_log_prob(segment)
                log_sums.append(alpha[j] + log_p_segment)
            if log_sums:
                alpha[i] = torch.logsumexp(torch.tensor(log_sums), dim=0).item()
        return alpha

    def _backward_pass(self, word):
        n = len(word)
        beta = [self.LOG_ZERO] * (n + 1)
        beta[n] = 0.0

        for i in range(n - 1, -1, -1):
            log_sums = []
            for j in range(i + 1, min(n + 1, i + self.max_len + 1)):
                segment = word[i:j]
                log_p_segment = self._get_log_prob(segment)
                log_sums.append(beta[j] + log_p_segment)
            if log_sums:
                beta[i] = torch.logsumexp(torch.tensor(log_sums), dim=0).item()
        return beta

    def get_boundary_priors(self, word):
        """Compute P(boundary at position i | word) for each position."""
        n = len(word)
        if n <= 1:
            return []

        alpha = self._forward_pass(word)
        beta = self._backward_pass(word)
        
        log_total_prob = alpha[n]
        if log_total_prob == self.LOG_ZERO:
            return [0.0] * (n - 1)

        log_priors = []
        for i in range(1, n):
            log_p_boundary = alpha[i] + beta[i]
            log_priors.append(log_p_boundary)
        
        log_priors_tensor = torch.tensor(log_priors)
        normalized_log_priors = log_priors_tensor - log_total_prob
        return torch.exp(normalized_log_priors).tolist()'''
    nb.cells.append(new_code_cell(hmm_class))
    
    hmm_training = '''def train_hmm_prior(samples):
    """Learn suffix probabilities from training segmentations."""
    suffix_counts = Counter()
    max_suffix_len = 0
    
    for s in samples:
        cs = s["tokens"]
        morph_lens = [len(seg) for seg in safe_list(s['y_morphs'])]
        
        current_idx = len(cs)
        for morph_len in reversed(morph_lens[1:]):
            start_idx = current_idx - morph_len
            suffix_tokens = cs[start_idx:current_idx]
            suffix_str = "".join(suffix_tokens)
            suffix_counts[suffix_str] += 1
            max_suffix_len = max(max_suffix_len, len(suffix_str))
            current_idx = start_idx

    total_suffix_obs = sum(suffix_counts.values())
    
    log_probs = {
        suffix: math.log((count + 1) / (total_suffix_obs + len(suffix_counts)))
        for suffix, count in suffix_counts.items()
    }

    avg_log_prob = sum(log_probs.values()) / len(log_probs) if log_probs else 0
    unk_penalty = avg_log_prob * 1.5

    print(f"HMM: {len(log_probs)} suffixes, max len {max_suffix_len}, unk penalty {unk_penalty:.2f}")
    return SuffixHMMPrior(log_probs, max_suffix_len, unk_penalty=unk_penalty)

def create_hmm_prior_from_list(allowed_suffixes: list, unk_penalty: float = -15.0):
    """Build HMM prior from a predefined suffix list instead of learning."""
    if not allowed_suffixes:
        raise ValueError("suffix list can't be empty")

    suffix_log_probs = {suffix: 0.0 for suffix in allowed_suffixes}
    max_suffix_len = len(max(allowed_suffixes, key=len))

    print(f"HMM: initialized with {len(allowed_suffixes)} suffixes, max len {max_suffix_len}")
    return SuffixHMMPrior(suffix_log_probs, max_suffix_len, unk_penalty=unk_penalty)'''
    nb.cells.append(new_code_cell(hmm_training))
    
    sample_builder = '''def build_samples_with_priv(df, feat_names=NEW_NUM_FEATS):
    """Convert dataframe rows to sample dicts for training."""
    rows = []
    for _, r in df.iterrows():
        cs = safe_list(r["Char_split"])
        toks = flatten(cs)
        lens = [len(seg) for seg in cs]
        cut_idxs = set(np.cumsum(lens)[:-1].tolist())
        y = [1 if (i + 1) in cut_idxs else 0 for i in range(len(toks) - 1)]
        priv = extract_priv_features_from_row(r, feat_names)
        gold_morphs = ["".join(seg) for seg in cs]
        rows.append({"tokens": toks, "y": y, "priv": priv, "y_morphs": gold_morphs})
    return rows'''
    nb.cells.append(new_code_cell(sample_builder))


def add_prior_functions_section(nb):
    """Add functions for computing priors."""
    
    prior_funcs = '''def featurize_window(tokens, i, k_left=2, k_right=2):
    """Extract local context features around position i."""
    feats = {}
    for k in range(1, k_left + 1):
        idx = i - (k - 1)
        feats[f"L{k}"] = tokens[idx] if idx >= 0 else "<BOS>"
    for k in range(1, k_right + 1):
        idx = i + k
        feats[f"R{k}"] = tokens[idx] if idx < len(tokens) else "<EOS>"
    
    def is_vowel(ch):
        return ch.lower() in "aeiouáéíóú"
    
    L1 = feats["L1"]
    R1 = feats["R1"]
    feats["L1_cv"] = 'V' if is_vowel(L1[-1]) else 'C'
    feats["R1_cv"] = 'V' if (R1 != "<EOS>" and is_vowel(R1[0])) else 'C'
    feats["L1_last"] = L1[-1]
    feats["R1_first"] = R1[0] if R1 != "<EOS>" else "<EOS>"
    return feats

def prior_probs_for_sample(hmm_prior, tokens):
    """Get HMM boundary priors mapped to token positions."""
    if hmm_prior is None or len(tokens) <= 1:
        return [0.5] * (max(len(tokens) - 1, 0))

    word = "".join(tokens)
    char_priors = hmm_prior.get_boundary_priors(word)

    # Map character-level to token-level
    token_boundary_indices = np.cumsum([len(t) for t in tokens[:-1]]) - 1
    
    token_priors = []
    for idx in token_boundary_indices:
        if 0 <= idx < len(char_priors):
            token_priors.append(char_priors[idx])
        else:
            token_priors.append(0.5)

    return token_priors'''
    nb.cells.append(new_code_cell(prior_funcs))


def add_k_teacher_section(nb):
    """Add K-teacher regularization code."""
    
    k_teacher = '''def train_k_teacher_priv(samples, feat_dim):
    """Train a regressor to predict number of cuts from features."""
    X = np.array([s["priv"] for s in samples], dtype=float)
    y = np.array([int(np.sum(s["y"])) for s in samples], dtype=float)
    reg = DecisionTreeRegressor(max_depth=6, min_samples_leaf=10, random_state=RNG)
    reg.fit(X, y)
    return reg

def predict_k_hat_priv(reg, priv_batch):
    """Predict expected number of cuts for a batch."""
    with torch.no_grad():
        k = reg.predict(priv_batch.cpu().numpy())
    return torch.tensor(k, dtype=torch.float32, device=priv_batch.device)'''
    nb.cells.append(new_code_cell(k_teacher))


def add_dataset_section(nb):
    """Add dataset and dataloader code."""
    
    vocab_code = '''def build_vocab(samples, min_freq=1):
    """Build token vocabulary from samples."""
    ctr = Counter()
    for s in samples:
        ctr.update(s["tokens"])
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for t, c in sorted(ctr.items(), key=lambda x: (-x[1], x[0])):
        if c >= min_freq and t not in vocab:
            vocab[t] = len(vocab)
    return vocab'''
    nb.cells.append(new_code_cell(vocab_code))
    
    dataset_code = '''class SegDataset(Dataset):
    """Dataset for boundary prediction training."""
    def __init__(self, samples, vocab, hmm_prior=None, feat_dim=0):
        self.samples = samples
        self.vocab = vocab
        self.hmm_prior = hmm_prior
        self.feat_dim = feat_dim

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        tokens = s["tokens"]
        ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]
        y = s["y"]
        prior = prior_probs_for_sample(self.hmm_prior, tokens)
        priv = s["priv"] if self.feat_dim > 0 else []
        return {"ids": ids, "y": y, "prior": prior, "priv": priv, "tokens": tokens}

def collate(batch):
    """Collate samples into batched tensors."""
    maxT = max(len(b["ids"]) for b in batch)
    maxB = maxT - 1
    B = len(batch)

    ids = torch.full((B, maxT), 0, dtype=torch.long)
    mask_tok = torch.zeros((B, maxT), dtype=torch.bool)
    y = torch.full((B, maxB), -100, dtype=torch.long)
    prior = torch.zeros((B, maxB), dtype=torch.float32)
    mask_b = torch.zeros((B, maxB), dtype=torch.bool)

    feat_dim = len(batch[0]["priv"]) if isinstance(batch[0]["priv"], list) else 0
    priv = torch.zeros((B, feat_dim), dtype=torch.float32) if feat_dim > 0 else None

    for i, b in enumerate(batch):
        T = len(b["ids"])
        ids[i, :T] = torch.tensor(b["ids"], dtype=torch.long)
        mask_tok[i, :T] = True
        if T > 1:
            L = T - 1
            y[i, :L] = torch.tensor(b["y"], dtype=torch.long)
            p = b["prior"] if len(b["prior"]) == L else [0.5] * L
            prior[i, :L] = torch.tensor(p, dtype=torch.float32)
            mask_b[i, :L] = True
        if feat_dim > 0:
            priv[i] = torch.tensor(b["priv"], dtype=torch.float32)

    return {
        "ids": ids, "mask_tok": mask_tok,
        "y": y, "prior": prior, "mask_b": mask_b,
        "priv": priv
    }'''
    nb.cells.append(new_code_cell(dataset_code))


def add_model_section(nb):
    """Add BiLSTM model definition."""
    
    model_code = '''class BiLSTMTagger(nn.Module):
    """
    Bidirectional LSTM for boundary prediction.
    Can fuse HMM prior via concatenation or logit addition.
    """
    def __init__(self, vocab_size, emb_dim=16, hidden_size=64, num_layers=2,
                 use_prior=True, dropout=0.1, freeze_emb=False, fuse_mode="logit_add"):
        super().__init__()
        self.use_prior = use_prior
        self.fuse_mode = fuse_mode
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if freeze_emb:
            for p in self.emb.parameters():
                p.requires_grad = False
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=emb_dim, hidden_size=hidden_size // 2,
            num_layers=num_layers, dropout=lstm_dropout,
            bidirectional=True, batch_first=True
        )
        in_mlp = hidden_size + (1 if (use_prior and fuse_mode == "concat") else 0)
        self.boundary_mlp = nn.Sequential(
            nn.Linear(in_mlp, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)
        )
        if use_prior and fuse_mode == "logit_add":
            self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, ids, prior, mask_tok):
        emb = self.emb(ids)
        h, _ = self.lstm(emb)
        left = h[:, :-1, :]
        if self.use_prior and self.fuse_mode == "concat":
            feat = torch.cat([left, prior.unsqueeze(-1)], dim=-1)
            return self.boundary_mlp(feat)
        logits = self.boundary_mlp(left)
        if self.use_prior and self.fuse_mode == "logit_add":
            eps = 1e-6
            p = prior.clamp(eps, 1 - eps)
            prior_logit = torch.log(p) - torch.log(1 - p)
            logits[..., 1] = logits[..., 1] + self.alpha * prior_logit
        return logits'''
    nb.cells.append(new_code_cell(model_code))


def add_training_section(nb):
    """Add training loop and utilities."""
    
    metrics_code = '''def boundary_metrics_from_lists(probs_list, gold_list, thr=0.5):
    """Compute precision/recall/F1 for boundary prediction."""
    if not probs_list:
        return 0.0, 0.0, 0.0
    p = torch.cat([t for t in probs_list if t.numel() > 0], dim=0).numpy()
    g = torch.cat([t for t in gold_list if t.numel() > 0], dim=0).numpy()
    pred = (p >= thr).astype(int)
    P, R, F1, _ = precision_recall_fscore_support(g, pred, average='binary', zero_division=0)
    return P, R, F1

def exact_match_rate_from_lists(probs_list, gold_list, thr=0.5):
    """Fraction of words with perfectly predicted boundaries."""
    if not probs_list:
        return 0.0
    em = []
    for p, g in zip(probs_list, gold_list):
        if g.numel() == 0:
            em.append(1.0)
        else:
            pred = (p.numpy() >= thr).astype(int)
            em.append(float(np.array_equal(pred, g.numpy())))
    return float(np.mean(em))

@torch.no_grad()
def predict(model, loader):
    """Run model on loader, return probability and gold lists."""
    model.eval()
    probs_list, gold_list = [], []
    for batch in loader:
        logits = model(batch["ids"], batch["prior"], batch["mask_tok"])
        probs = torch.softmax(logits, dim=-1)[..., 1]
        y = batch["y"]
        mask = batch["mask_b"]
        B = probs.shape[0]
        for b in range(B):
            L = int(mask[b].sum().item())
            if L == 0:
                probs_list.append(torch.empty(0))
                gold_list.append(torch.empty(0, dtype=torch.long))
            else:
                probs_list.append(probs[b, :L].cpu())
                gold_list.append(y[b, :L].cpu())
    return probs_list, gold_list'''
    nb.cells.append(new_code_cell(metrics_code))
    
    train_code = '''criterion_ce = nn.CrossEntropyLoss()
criterion_bce = nn.BCEWithLogitsLoss(reduction="mean")
mse = nn.MSELoss(reduction="mean")

def train_epoch(model, loader, opt, lambda_prior=0.1, lambda_k=0.1, k_reg=None):
    """One training epoch with optional prior distillation and K regularization."""
    model.train()
    tot = 0
    n = 0
    for batch in loader:
        ids, prior, y, mask_b = batch["ids"], batch["prior"], batch["y"], batch["mask_b"]
        priv = batch["priv"]

        logits = model(ids, prior, batch["mask_tok"])
        logits_flat = logits[mask_b]
        y_true = y[mask_b]

        # Main CE loss
        loss = criterion_ce(logits_flat, y_true)

        # Prior distillation
        if lambda_prior > 0:
            cut_logit = logits[..., 1]
            prior_flat = prior[mask_b]
            loss_pr = criterion_bce(cut_logit[mask_b], prior_flat)
            loss = loss + lambda_prior * loss_pr

        # K regularization
        if (lambda_k > 0) and (k_reg is not None) and (priv is not None):
            with torch.no_grad():
                k_hat = predict_k_hat_priv(k_reg, priv)
            cut_logit = logits[..., 1]
            p_cut = torch.sigmoid(cut_logit)
            exp_K = p_cut.sum(dim=1)
            loss_k = mse(exp_K, k_hat)
            loss = loss + lambda_k * loss_k

        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item()
        n += 1
    return tot / max(n, 1)

def split_train_test(samples, test_ratio=0.2):
    """Random train/test split."""
    n = len(samples)
    idx = np.arange(n)
    np.random.shuffle(idx)
    cut = int(n * (1 - test_ratio))
    tr = [samples[i] for i in idx[:cut]]
    te = [samples[i] for i in idx[cut:]]
    return tr, te

def best_threshold_for_exact(probs_list, gold_list, grid=None):
    """Find threshold that maximizes exact match rate."""
    if grid is None:
        grid = np.linspace(0.3, 0.9, 61)
    best_thr, best_em, best_f1 = 0.5, -1.0, 0.0
    p_all = np.concatenate([t.numpy() for t in probs_list if t.numel() > 0], axis=0)
    g_all = np.concatenate([t.numpy() for t in gold_list if t.numel() > 0], axis=0)
    for thr in grid:
        ems = []
        for p, g in zip(probs_list, gold_list):
            if g.numel() == 0:
                ems.append(1.0)
                continue
            ems.append(float(np.array_equal((p.numpy() >= thr).astype(int), g.numpy())))
        em = float(np.mean(ems))
        pred_all = (p_all >= thr).astype(int)
        P, R, F1, _ = precision_recall_fscore_support(g_all, pred_all, average='binary', zero_division=0)
        if em > best_em or (np.isclose(em, best_em) and F1 > best_f1):
            best_thr, best_em, best_f1 = thr, em, F1
    print(f"best threshold: {best_thr:.3f} | exact={best_em:.3f} | F1={best_f1:.3f}")
    return best_thr'''
    nb.cells.append(new_code_cell(train_code))


def add_model_io_section(nb):
    """Add model saving and loading functions."""
    
    io_code = '''def generate_model_id(df, provided_suffix_list, use_suffix_list, unk_penalty, epochs,
                      use_prior, fuse_mode, lambda_prior, lambda_k, batch_size, hparams, synthetic_choice,
                      augmentation_word_selection=None, augmentation_n_words=None):
    """Hash training params to get unique model ID."""
    if augmentation_word_selection is None:
        augmentation_word_selection = globals().get('AUGMENTATION_WORD_SELECTION', 'all')
    if augmentation_n_words is None:
        augmentation_n_words = globals().get('AUGMENTATION_N_WORDS', None)
    
    params_dict = {
        'synthetic_choice': synthetic_choice,
        'use_suffix_list': use_suffix_list,
        'unk_penalty': unk_penalty,
        'epochs': epochs,
        'use_prior': use_prior,
        'fuse_mode': fuse_mode,
        'lambda_prior': lambda_prior,
        'lambda_k': lambda_k,
        'batch_size': batch_size,
        'hparams': hparams,
        'suffix_list_len': len(provided_suffix_list) if provided_suffix_list else 0,
        'df_shape': df.shape if df is not None else (0, 0),
        'augmentation_word_selection': augmentation_word_selection,
        'augmentation_n_words': augmentation_n_words
    }
    params_str = json.dumps(params_dict, sort_keys=True)
    model_id = hashlib.md5(params_str.encode()).hexdigest()[:16]
    return model_id

def save_model(model, vocab, out, model_id, models_folder=MODELS_FOLDER,
               synthetic_choice=None, augmentation_word_selection=None, augmentation_n_words=None):
    """Save model weights and artifacts."""
    model_dir = os.path.join(models_folder, model_id)
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
    
    with open(os.path.join(model_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    
    with open(os.path.join(model_dir, "artifacts.pkl"), "wb") as f:
        pickle.dump(out, f)
    
    if synthetic_choice is None:
        synthetic_choice = globals().get('SYNTHETIC_DATA_CHOICE', 'none')
    if augmentation_word_selection is None:
        augmentation_word_selection = globals().get('AUGMENTATION_WORD_SELECTION', 'all')
    if augmentation_n_words is None:
        augmentation_n_words = globals().get('AUGMENTATION_N_WORDS', None)
    
    metadata = {
        'model_id': model_id,
        'vocab_size': len(vocab),
        'synthetic_choice': synthetic_choice,
        'augmentation_word_selection': augmentation_word_selection,
    }
    if augmentation_n_words is not None:
        metadata['augmentation_n_words'] = augmentation_n_words
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"saved model to {model_dir}")
    return model_dir

def load_model(model_id, models_folder=MODELS_FOLDER, vocab_size=None):
    """Load saved model artifacts."""
    model_dir = os.path.join(models_folder, model_id)
    if not os.path.exists(model_dir):
        return None
    
    vocab_path = os.path.join(model_dir, "vocab.pkl")
    if not os.path.exists(vocab_path):
        return None
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    
    artifacts_path = os.path.join(model_dir, "artifacts.pkl")
    if not os.path.exists(artifacts_path):
        return None
    with open(artifacts_path, "rb") as f:
        out = pickle.load(f)
    
    model_path = os.path.join(model_dir, "model.pt")
    if not os.path.exists(model_path):
        return None
    
    print(f"loaded artifacts from {model_dir}")
    return {
        'vocab': vocab,
        'out': out,
        'model_state_path': model_path,
        'model_dir': model_dir
    }'''
    nb.cells.append(new_code_cell(io_code))


def add_main_training_function(nb):
    """Add the main training orchestration function."""
    
    main_train = '''def run_segmentation_with_privK(
    df,
    provided_suffix_list,
    use_suffix_list=True,
    unk_penalty=-15.0,
    epochs=15,
    use_prior=True,
    fuse_mode="logit_add",
    lambda_prior=0.1,
    lambda_k=0.2,
    batch_size=64,
    hparams=None,
    synthetic_choice=None
):
    """
    Train or load a segmentation model. Checks for existing checkpoints first.
    """
    if hparams is None:
        hparams = dict(emb_dim=16, hidden_size=64, num_layers=2,
                       dropout=0.25, lr=1e-3, weight_decay=1e-4, freeze_emb=False)
    
    if synthetic_choice is None:
        synthetic_choice = SYNTHETIC_DATA_CHOICE if 'SYNTHETIC_DATA_CHOICE' in globals() else "none"
    
    augmentation_word_selection = globals().get('AUGMENTATION_WORD_SELECTION', 'all')
    augmentation_n_words = globals().get('AUGMENTATION_N_WORDS', None)
    
    model_id = generate_model_id(
        df, provided_suffix_list, use_suffix_list, unk_penalty, epochs,
        use_prior, fuse_mode, lambda_prior, lambda_k, batch_size, hparams, synthetic_choice,
        augmentation_word_selection=augmentation_word_selection,
        augmentation_n_words=augmentation_n_words
    )
    
    print(f"looking for model {model_id}...")
    loaded = load_model(model_id, models_folder=MODELS_FOLDER)
    
    if loaded is not None:
        print(f"found it! loading from {loaded['model_dir']}")
        vocab = loaded['vocab']
        out = loaded['out']
        model_state_path = loaded['model_state_path']
        
        model = BiLSTMTagger(
            vocab_size=len(vocab),
            emb_dim=hparams.get("emb_dim", 16),
            hidden_size=hparams.get("hidden_size", 64),
            num_layers=hparams.get("num_layers", 2),
            use_prior=(use_prior and fuse_mode != "none"),
            dropout=hparams.get("dropout", 0.25),
            freeze_emb=hparams.get("freeze_emb", False),
            fuse_mode=fuse_mode
        )
        model.load_state_dict(torch.load(model_state_path))
        model.eval()
        print("skipping training, model ready")
        return model, vocab, out
    
    print(f"no checkpoint found, training from scratch...")
    
    samples = build_samples_with_priv(df, feat_names=NEW_NUM_FEATS)
    train_s, test_s = split_train_test(samples, 0.2)

    hmm_prior = None
    if use_prior and use_suffix_list:
        hmm_prior = create_hmm_prior_from_list(provided_suffix_list, unk_penalty)
    if use_prior and not use_suffix_list:
        hmm_prior = train_hmm_prior(train_s)

    feat_dim = len(NEW_NUM_FEATS)
    k_reg = train_k_teacher_priv(train_s, feat_dim=feat_dim)

    vocab = build_vocab(train_s, min_freq=1)

    train_ds = SegDataset(train_s, vocab, hmm_prior=hmm_prior, feat_dim=feat_dim)
    test_ds = SegDataset(test_s, vocab, hmm_prior=hmm_prior, feat_dim=feat_dim)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = BiLSTMTagger(
        vocab_size=len(vocab),
        emb_dim=hparams.get("emb_dim", 16),
        hidden_size=hparams.get("hidden_size", 64),
        num_layers=hparams.get("num_layers", 2),
        use_prior=(use_prior and fuse_mode != "none"),
        dropout=hparams.get("dropout", 0.25),
        freeze_emb=hparams.get("freeze_emb", False),
        fuse_mode=fuse_mode
    )

    opt = torch.optim.AdamW(model.parameters(), lr=hparams.get("lr", 1e-3), 
                            weight_decay=hparams.get("weight_decay", 1e-4))

    final_probs_list, final_gold_list = None, None
    for ep in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, opt, lambda_prior=lambda_prior, lambda_k=lambda_k, k_reg=k_reg)
        probs_list, gold_list = predict(model, test_loader)
        P, R, F1 = boundary_metrics_from_lists(probs_list, gold_list, thr=0.5)
        EM = exact_match_rate_from_lists(probs_list, gold_list, thr=0.5)
        print(f"epoch {ep:02d} | loss={loss:.4f} | P/R/F1={P:.3f}/{R:.3f}/{F1:.3f} | exact={EM:.3f}")
        final_probs_list, final_gold_list = probs_list, gold_list

    best_thr = best_threshold_for_exact(final_probs_list, final_gold_list)

    out = {
        "probs_list": final_probs_list,
        "gold_list": final_gold_list,
        "hmm_prior": hmm_prior,
        "k_teacher": k_reg,
        "best_thr": best_thr
    }
    
    print(f"saving model {model_id}...")
    save_model(model, vocab, out, model_id, models_folder=MODELS_FOLDER,
               synthetic_choice=synthetic_choice,
               augmentation_word_selection=augmentation_word_selection,
               augmentation_n_words=augmentation_n_words)

    return model, vocab, out'''
    nb.cells.append(new_code_cell(main_train))


def add_kfold_section(nb):
    """Add k-fold cross-validation function."""
    
    kfold_code = '''def run_kfold_cross_validation(
    df,
    provided_suffix_list,
    n_folds=5,
    use_suffix_list=True,
    unk_penalty=-15.0,
    epochs=15,
    use_prior=True,
    fuse_mode="logit_add",
    lambda_prior=0.1,
    lambda_k=0.2,
    batch_size=64,
    hparams=None,
    synthetic_choice=None,
    random_state=42
):
    """K-fold cross-validation for more robust evaluation."""
    if hparams is None:
        hparams = dict(emb_dim=16, hidden_size=64, num_layers=2,
                       dropout=0.25, lr=1e-3, weight_decay=1e-4, freeze_emb=False)
    
    if synthetic_choice is None:
        synthetic_choice = SYNTHETIC_DATA_CHOICE if 'SYNTHETIC_DATA_CHOICE' in globals() else "none"
    
    print(f"\\n{'=' * 80}")
    print(f"K-FOLD CV (k={n_folds})")
    print(f"{'=' * 80}")
    
    samples = build_samples_with_priv(df, feat_names=NEW_NUM_FEATS)
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    fold_results = []
    all_metrics = {
        'boundary_precision': [],
        'boundary_recall': [],
        'boundary_f1': [],
        'exact_match': [],
        'best_threshold': []
    }
    
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(samples), 1):
        print(f"\\n--- fold {fold_idx}/{n_folds} ---")
        print(f"train: {len(train_indices)}, val: {len(val_indices)}")
        
        train_samples = [samples[i] for i in train_indices]
        val_samples = [samples[i] for i in val_indices]
        
        hmm_prior = None
        if use_prior and use_suffix_list:
            hmm_prior = create_hmm_prior_from_list(provided_suffix_list, unk_penalty)
        elif use_prior and not use_suffix_list:
            hmm_prior = train_hmm_prior(train_samples)
        
        feat_dim = len(NEW_NUM_FEATS)
        k_reg = train_k_teacher_priv(train_samples, feat_dim=feat_dim)
        vocab = build_vocab(train_samples, min_freq=1)
        
        train_ds = SegDataset(train_samples, vocab, hmm_prior=hmm_prior, feat_dim=feat_dim)
        val_ds = SegDataset(val_samples, vocab, hmm_prior=hmm_prior, feat_dim=feat_dim)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
        
        model = BiLSTMTagger(
            vocab_size=len(vocab),
            emb_dim=hparams.get("emb_dim", 16),
            hidden_size=hparams.get("hidden_size", 64),
            num_layers=hparams.get("num_layers", 2),
            use_prior=(use_prior and fuse_mode != "none"),
            dropout=hparams.get("dropout", 0.25),
            freeze_emb=hparams.get("freeze_emb", False),
            fuse_mode=fuse_mode
        )
        
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=hparams.get("lr", 1e-3),
            weight_decay=hparams.get("weight_decay", 1e-4)
        )
        
        best_val_em = -1.0
        best_val_f1 = -1.0
        best_epoch = 0
        
        for ep in range(1, epochs + 1):
            loss = train_epoch(model, train_loader, opt, lambda_prior=lambda_prior, lambda_k=lambda_k, k_reg=k_reg)
            probs_list, gold_list = predict(model, val_loader)
            P, R, F1 = boundary_metrics_from_lists(probs_list, gold_list, thr=0.5)
            EM = exact_match_rate_from_lists(probs_list, gold_list, thr=0.5)
            
            print(f"  ep {ep:02d} | loss={loss:.4f} | P/R/F1={P:.3f}/{R:.3f}/{F1:.3f} | exact={EM:.3f}")
            
            if EM > best_val_em or (np.isclose(EM, best_val_em) and F1 > best_val_f1):
                best_val_em = EM
                best_val_f1 = F1
                best_epoch = ep
                best_probs_list = probs_list
                best_gold_list = gold_list
        
        best_thr = best_threshold_for_exact(best_probs_list, best_gold_list)
        P_final, R_final, F1_final = boundary_metrics_from_lists(best_probs_list, best_gold_list, thr=best_thr)
        EM_final = exact_match_rate_from_lists(best_probs_list, best_gold_list, thr=best_thr)
        
        print(f"  best epoch: {best_epoch}")
        print(f"  final (thr={best_thr:.3f}): P/R/F1={P_final:.3f}/{R_final:.3f}/{F1_final:.3f} | exact={EM_final:.3f}")
        
        fold_results.append({
            'fold': fold_idx,
            'boundary_precision': P_final,
            'boundary_recall': R_final,
            'boundary_f1': F1_final,
            'exact_match': EM_final,
            'best_threshold': best_thr,
            'best_epoch': best_epoch
        })
        
        all_metrics['boundary_precision'].append(P_final)
        all_metrics['boundary_recall'].append(R_final)
        all_metrics['boundary_f1'].append(F1_final)
        all_metrics['exact_match'].append(EM_final)
        all_metrics['best_threshold'].append(best_thr)
    
    mean_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    std_metrics = {k: np.std(v) for k, v in all_metrics.items()}
    best_fold_idx = max(range(len(fold_results)), key=lambda i: fold_results[i]['exact_match'])
    
    print(f"\\n{'=' * 80}")
    print("CV SUMMARY")
    print(f"{'=' * 80}")
    for r in fold_results:
        print(f"  fold {r['fold']}: P={r['boundary_precision']:.3f}, R={r['boundary_recall']:.3f}, "
              f"F1={r['boundary_f1']:.3f}, EM={r['exact_match']:.3f}")
    
    print(f"\\nmean +/- std over {n_folds} folds:")
    print(f"  precision: {mean_metrics['boundary_precision']:.3f} +/- {std_metrics['boundary_precision']:.3f}")
    print(f"  recall:    {mean_metrics['boundary_recall']:.3f} +/- {std_metrics['boundary_recall']:.3f}")
    print(f"  F1:        {mean_metrics['boundary_f1']:.3f} +/- {std_metrics['boundary_f1']:.3f}")
    print(f"  exact:     {mean_metrics['exact_match']:.3f} +/- {std_metrics['exact_match']:.3f}")
    print(f"  threshold: {mean_metrics['best_threshold']:.3f} +/- {std_metrics['best_threshold']:.3f}")
    print(f"\\nbest fold: {fold_results[best_fold_idx]['fold']} (exact={fold_results[best_fold_idx]['exact_match']:.3f})")
    print(f"{'=' * 80}\\n")
    
    return {
        'fold_results': fold_results,
        'mean_metrics': mean_metrics,
        'std_metrics': std_metrics,
        'best_fold_idx': best_fold_idx,
        'all_metrics': all_metrics
    }'''
    nb.cells.append(new_code_cell(kfold_code))


def add_inference_section(nb):
    """Add inference and tokenization functions."""
    
    inference_code = '''def tokenize_with_vocab(word: str, vocab: dict, max_token_len: int = 4):
    """Greedy left-to-right tokenization using vocab."""
    i, toks = 0, []
    while i < len(word):
        matched = None
        Lmax = min(max_token_len, len(word) - i)
        for L in range(Lmax, 0, -1):
            seg = word[i:i + L]
            if seg in vocab:
                matched = seg
                break
        toks.append(matched if matched else word[i])
        i += len(toks[-1])
    return toks

@torch.no_grad()
def segment_tokens(model, vocab, tokens, hmm_prior=None, thr=0.5):
    """Segment a tokenized word and return the segmented string + probabilities."""
    ids = torch.tensor([[vocab.get(t, vocab["<UNK>"]) for t in tokens]], dtype=torch.long)
    mask_tok = torch.ones_like(ids, dtype=torch.bool)
    T = len(tokens)
    if T <= 1:
        return "".join(tokens), np.array([])
    
    prior_list = prior_probs_for_sample(hmm_prior, tokens)
    prior = torch.tensor([prior_list], dtype=torch.float32)
    logits = model(ids, prior, mask_tok)
    probs = torch.softmax(logits, dim=-1)[0, :, 1].cpu().numpy()
    cuts = (probs >= thr).astype(int)
    
    out = []
    for i, tok in enumerate(tokens):
        out.append(tok)
        if i < T - 1 and cuts[i] == 1:
            out.append("-")
    return "".join(out), probs'''
    nb.cells.append(new_code_cell(inference_code))


def add_evaluation_section(nb):
    """Add evaluation functions."""
    
    eval_helpers = '''def offsets_from_morphemes(morphs: List[str]) -> Set[int]:
    """Character offsets of boundaries between morphemes."""
    offs = []
    s = 0
    for i, m in enumerate(morphs):
        s += len(m)
        if i < len(morphs) - 1:
            offs.append(s)
    return set(offs)

def offsets_from_tokens_and_mask(tokens: List[str], mask01: np.ndarray) -> Set[int]:
    """Character offsets where model predicted boundaries."""
    offs = set()
    cum = 0
    for i, t in enumerate(tokens):
        cum += len(t)
        if i < len(tokens) - 1 and mask01[i] == 1:
            offs.add(cum)
    return offs

def f1_from_sets(pred: Set[int], gold: Set[int]) -> Tuple[float, float, float, int, int, int]:
    """P/R/F1 from predicted and gold boundary sets."""
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return P, R, F1, tp, fp, fn

def normalize_gold_variants(gold_variants):
    """Convert gold variants to proper list format."""
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
    return []'''
    nb.cells.append(new_code_cell(eval_helpers))
    
    eval_main = '''def evaluate_on_gold_df(df, model, vocab, out, max_token_len=4, use_tuned_thr=True, show_sample=5):
    """Evaluate model on test set with multiple gold variants per word."""
    hmm_prior = out["hmm_prior"]
    thr = float(out.get("best_thr", 0.5)) if use_tuned_thr else 0.5

    total_tp = total_fp = total_fn = 0
    exact_hits = 0
    n_eval = 0
    examples = []

    for _, row in df.iterrows():
        word = str(row["Word"])
        gold_variants = normalize_gold_variants(row["Gold"])

        if not isinstance(gold_variants, list) or len(gold_variants) == 0:
            continue

        toks = tokenize_with_vocab(word, vocab, max_token_len=max_token_len)
        seg_string, probs = segment_tokens(model, vocab, toks, hmm_prior=hmm_prior, thr=thr)
        mask01 = (probs >= thr).astype(int)
        pred_set = offsets_from_tokens_and_mask(toks, mask01)

        gold_sets = [offsets_from_morphemes(gv) for gv in gold_variants]

        if any(pred_set == gs for gs in gold_sets):
            exact_hits += 1

        best = max((f1_from_sets(pred_set, gs) + (gs,) for gs in gold_sets), key=lambda z: z[2])
        P, R, F1, tp, fp, fn, best_gs = best

        total_tp += tp
        total_fp += fp
        total_fn += fn
        n_eval += 1

        if len(examples) < show_sample:
            best_morphs = None
            for gv in gold_variants:
                if offsets_from_morphemes(gv) == best_gs:
                    best_morphs = gv
                    break
            gold_str = "-".join(best_morphs) if best_morphs else "(ambig)"
            examples.append({
                "word": word, "tokens": toks, "pred_seg": seg_string,
                "gold_best": gold_str, "P": round(P, 3), "R": round(R, 3), "F1": round(F1, 3)
            })

    micro_P = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_R = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_F1 = 2 * micro_P * micro_R / (micro_P + micro_R) if (micro_P + micro_R) > 0 else 0.0
    exact_rate = exact_hits / n_eval if n_eval > 0 else 0.0

    print(f"evaluated {n_eval} words")
    print(f"boundary (micro) P/R/F1 = {micro_P:.3f}/{micro_R:.3f}/{micro_F1:.3f}")
    print(f"exact match = {exact_rate:.3f}")
    if examples:
        print("\\nsamples:")
        for ex in examples:
            print(f"- {ex['word']}")
            print(f"  tokens: {ex['tokens']}")
            print(f"  pred:   {ex['pred_seg']}")
            print(f"  gold:   {ex['gold_best']}")
            print(f"  P/R/F1: {ex['P']}/{ex['R']}/{ex['F1']}\\n")

    return {
        "n_eval": n_eval, "micro_precision": micro_P, "micro_recall": micro_R,
        "micro_f1": micro_F1, "exact_match_rate": exact_rate, "examples": examples
    }'''
    nb.cells.append(new_code_cell(eval_main))


def add_suffix_validation_section(nb):
    """Add suffix validation and filtered evaluation."""
    
    validation_code = '''def is_segmentation_valid(segmentation: list, allowed_suffixes: set) -> bool:
    """Check if all suffixes (non-root morphemes) are in the allowed set."""
    if len(segmentation) <= 1:
        return True
    for morpheme in segmentation[1:]:
        if morpheme not in allowed_suffixes:
            return False
    return True

def evaluate_and_ignore_rejected(
    df, model, vocab, out,
    allowed_suffixes: list,
    max_token_len=4,
    use_tuned_thr=True,
    show_sample=5
):
    """Evaluate but skip predictions with invalid suffixes."""
    hmm_prior = out["hmm_prior"]
    thr = float(out.get("best_thr", 0.5)) if use_tuned_thr else 0.5
    allowed_suffixes_set = set(allowed_suffixes)

    total_tp = total_fp = total_fn = 0
    exact_hits = 0
    n_total_words = 0
    n_evaluated_words = 0
    rejection_count = 0
    false_rejection_count = 0
    correct_kept_count = 0
    examples = []

    for _, row in df.iterrows():
        word = str(row["Word"])
        gold_variants = normalize_gold_variants(row["Gold"])

        if not isinstance(gold_variants, list) or len(gold_variants) == 0:
            continue
        
        n_total_words += 1

        toks = tokenize_with_vocab(word, vocab, max_token_len=max_token_len)
        seg_string, probs = segment_tokens(model, vocab, toks, hmm_prior=hmm_prior, thr=thr)
        predicted_morphs = seg_string.split('-')
        
        mask01 = (probs >= thr).astype(int)
        pred_set = offsets_from_tokens_and_mask(toks, mask01)
        gold_sets = [offsets_from_morphemes(gv) for gv in gold_variants]
        is_correct = any(pred_set == gs for gs in gold_sets)

        if not is_segmentation_valid(predicted_morphs, allowed_suffixes_set):
            rejection_count += 1
            if is_correct:
                false_rejection_count += 1
            continue

        n_evaluated_words += 1
        
        if is_correct:
            correct_kept_count += 1
            exact_hits += 1

        best = max((f1_from_sets(pred_set, gs) + (gs,) for gs in gold_sets), key=lambda z: z[2])
        P, R, F1, tp, fp, fn, best_gs = best

        total_tp += tp
        total_fp += fp
        total_fn += fn

        if len(examples) < show_sample:
            best_morphs = None
            for gv in gold_variants:
                if offsets_from_morphemes(gv) == best_gs:
                    best_morphs = gv
                    break
            gold_str = "-".join(best_morphs) if best_morphs else "(ambig)"
            examples.append({
                "word": word, "tokens": toks, "pred_seg": seg_string,
                "gold_best": gold_str, "P": round(P, 3), "R": round(R, 3), "F1": round(F1, 3)
            })

    micro_P = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_R = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_F1 = 2 * micro_P * micro_R / (micro_P + micro_R) if (micro_P + micro_R) > 0 else 0.0
    exact_rate = exact_hits / n_evaluated_words if n_evaluated_words > 0 else 0.0
    
    filter_precision = correct_kept_count / n_evaluated_words if n_evaluated_words > 0 else 0.0
    total_correct = correct_kept_count + false_rejection_count
    false_rejection_rate = false_rejection_count / total_correct if total_correct > 0 else 0.0

    print(f"tried {n_total_words} words")
    print(f"rejected {rejection_count} ({rejection_count/n_total_words:.1%}) with invalid suffixes")
    print(f"scoring {n_evaluated_words} valid predictions")
    print(f"\\n--- filter analysis ---")
    print(f"filter precision: {filter_precision:.1%}")
    print(f"false rejection rate: {false_rejection_rate:.1%}")
    print(f"  correct kept: {correct_kept_count}")
    print(f"  correct rejected: {false_rejection_count}")
    print(f"  total correct: {total_correct}")
    print(f"\\n--- final scores (valid predictions only) ---")
    print(f"boundary (micro) P/R/F1 = {micro_P:.3f}/{micro_R:.3f}/{micro_F1:.3f}")
    print(f"exact match = {exact_rate:.3f}")

    if examples:
        print("\\nsamples:")
        for ex in examples:
            print(f"- {ex['word']}")
            print(f"  tokens: {ex['tokens']}")
            print(f"  pred:   {ex['pred_seg']}")
            print(f"  gold:   {ex['gold_best']}")
            print(f"  P/R/F1: {ex['P']}/{ex['R']}/{ex['F1']}\\n")
    
    return {
        "micro_f1": micro_F1, "exact_match_rate": exact_rate,
        "rejection_count": rejection_count, "false_rejection_count": false_rejection_count,
        "filter_precision": filter_precision, "false_rejection_rate": false_rejection_rate
    }'''
    nb.cells.append(new_code_cell(validation_code))


def add_suffix_loading_section(nb):
    """Add suffix list loading."""
    
    suffix_load = '''def read_suffixes(filename):
    """Read suffix list from file (format: 'number -suffix')."""
    suffixes = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                _, suffix = parts
                suffixes.append(suffix[1:])  # drop leading dash
    return suffixes

suffix_filename = os.path.join(DATA_FOLDER, "suffixesCQ-Anettte-Rios_LS.txt")
if not os.path.exists(suffix_filename):
    suffix_filename = "suffixesCQ-Anettte-Rios_LS.txt"
    if not os.path.exists(suffix_filename):
        print(f"warning: suffix file not found")
        suffix_list = []
    else:
        suffix_list = read_suffixes(suffix_filename)
        print(f"loaded {len(suffix_list)} suffixes from {suffix_filename}")
else:
    suffix_list = read_suffixes(suffix_filename)
    print(f"loaded {len(suffix_list)} suffixes from {suffix_filename}")'''
    nb.cells.append(new_code_cell(suffix_load))


def add_optuna_section(nb):
    """Add hyperparameter tuning code (commented)."""
    
    optuna_code = '''import optuna

def objective(trial: optuna.Trial) -> float:
    """Optuna objective for hyperparameter search."""
    hparams = {
        "emb_dim": trial.suggest_categorical("emb_dim", [16, 32, 64]),
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.05),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "freeze_emb": False,
    }
    lambda_prior_val = trial.suggest_float("lambda_prior", 0.0, 0.5)

    print(f"\\n--- trial {trial.number} ---")
    model, vocab, out = run_segmentation_with_privK(
        df=str_df,
        provided_suffix_list=suffix_list,
        use_suffix_list=True,
        unk_penalty=-15,
        epochs=15,
        use_prior=True,
        lambda_prior=lambda_prior_val,
        lambda_k=0.2,
        hparams=hparams,
        synthetic_choice=SYNTHETIC_DATA_CHOICE
    )

    test_set_results = evaluate_on_gold_df(
        df=acc_df, model=model, vocab=vocab, out=out,
        max_token_len=4, use_tuned_thr=True, show_sample=0
    )
    test_exact_match = test_set_results["exact_match_rate"]
    
    print(f"trial {trial.number} done | exact match: {test_exact_match:.4f}")
    return test_exact_match

# Uncomment to run hyperparameter search:
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=50)
# print(f"best params: {study.best_trial.params}")'''
    nb.cells.append(new_code_cell(optuna_code))


def add_best_hparams_section(nb):
    """Add best hyperparameters from tuning."""
    
    best_hparams = '''# Best hyperparameters from optuna search
best = {
    "emb_dim": 32,
    "hidden_size": 128,
    "num_layers": 3,
    "dropout": 0.4,
    "lr": 0.009213045798657327,
    "weight_decay": 0.0001132283214088801,
    "freeze_emb": False,
}'''
    nb.cells.append(new_code_cell(best_hparams))


def add_run_kfold_section(nb):
    """Add k-fold CV execution."""
    
    run_kfold = '''# Run k-fold cross-validation
kfold_results = run_kfold_cross_validation(
    df=str_df,
    provided_suffix_list=suffix_list,
    n_folds=5,
    use_suffix_list=False,
    unk_penalty=-15.0,
    epochs=15,
    use_prior=True,
    lambda_prior=0.15289202508573396,
    lambda_k=0.2,
    hparams=best,
    synthetic_choice=SYNTHETIC_DATA_CHOICE,
    random_state=RNG
)

print(f"\\navg exact match: {kfold_results['mean_metrics']['exact_match']:.3f} +/- {kfold_results['std_metrics']['exact_match']:.3f}")
print(f"avg boundary F1: {kfold_results['mean_metrics']['boundary_f1']:.3f} +/- {kfold_results['std_metrics']['boundary_f1']:.3f}")'''
    nb.cells.append(new_code_cell(run_kfold))


def add_train_model_section(nb):
    """Add actual model training/loading cell."""
    
    train_model = '''# Train or load the model (this creates model, vocab, out for later use)
model, vocab, out = run_segmentation_with_privK(
    df=str_df,
    provided_suffix_list=suffix_list,
    use_suffix_list=False,
    unk_penalty=-15.0,
    epochs=15,
    use_prior=True,
    lambda_prior=0.15289202508573396,
    lambda_k=0.2,
    hparams=best,
    synthetic_choice=SYNTHETIC_DATA_CHOICE
)

thr = out.get("best_thr", 0.5)
print(f"\\nmodel ready, threshold: {thr:.3f}")'''
    nb.cells.append(new_code_cell(train_model))


def add_example_usage_section(nb):
    """Add example word segmentation."""
    
    example = '''# Example segmentation
word = "pikunas"
tokens = tokenize_with_vocab(word, vocab, max_token_len=4)
thr = out.get("best_thr", 0.5)

seg_string, boundary_probs = segment_tokens(
    model, vocab, tokens, hmm_prior=out["hmm_prior"], thr=thr
)

print(f"word: {word}")
print(f"tokens: {tokens}")
print(f"probs: {np.round(boundary_probs, 3).tolist()}")
print(f"segmented (thr={thr:.3f}): {seg_string}")'''
    nb.cells.append(new_code_cell(example))


def add_final_eval_section(nb):
    """Add final evaluation calls."""
    
    final_eval = '''# Evaluate on test set
print("\\n--- standard evaluation ---")
results = evaluate_on_gold_df(
    acc_df, model, vocab, out,
    max_token_len=4,
    use_tuned_thr=True,
    show_sample=8
)'''
    nb.cells.append(new_code_cell(final_eval))
    
    filtered_eval = '''# Evaluate with suffix filter
print("\\n--- evaluation with suffix filter ---")
results_filtered = evaluate_and_ignore_rejected(
    acc_df, model, vocab, out,
    allowed_suffixes=suffix_list,
    show_sample=8
)'''
    nb.cells.append(new_code_cell(filtered_eval))


def main():
    """Build and save the refactored notebook."""
    nb = create_notebook()
    
    # Add sections in order
    add_header_section(nb)
    add_imports_section(nb)
    add_config_section(nb)
    add_data_loading_section(nb)
    add_preprocessing_section(nb)
    add_helper_functions_section(nb)
    add_hmm_prior_section(nb)
    add_prior_functions_section(nb)
    add_k_teacher_section(nb)
    add_dataset_section(nb)
    add_model_section(nb)
    add_training_section(nb)
    add_model_io_section(nb)
    add_main_training_function(nb)
    add_kfold_section(nb)
    add_inference_section(nb)
    add_evaluation_section(nb)
    add_suffix_validation_section(nb)
    add_suffix_loading_section(nb)
    add_optuna_section(nb)
    add_best_hparams_section(nb)
    add_run_kfold_section(nb)
    add_train_model_section(nb)
    add_example_usage_section(nb)
    add_final_eval_section(nb)
    
    # Save notebook
    output_path = "Markov-LSTM-MarkovFilter_refactored.ipynb"
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print(f"Refactored notebook saved to: {output_path}")


if __name__ == "__main__":
    main()

