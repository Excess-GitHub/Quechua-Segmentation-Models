#!/usr/bin/env python3
"""
Script to generate a refactored version of the DT-LSTM-MarkovFilter notebook.
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
        "# DT-LSTM-Markov Filter: Quechua Morphology Parser\n\n"
        "Morphological segmentation for Quechua using:\n"
        "- BiLSTM for boundary prediction\n"
        "- Decision Tree priors from token-window features\n"
        "- K-teacher regularization\n\n"
        "Unlike the HMM variant, this uses Decision Trees trained on local context features."
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
MODEL_NAME = "DT-LSTM-MarkovFilter"
MODELS_FOLDER = f"models_{MODEL_NAME}"
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Random seeds
RANDOM_STATE = 42
RNG = 42
torch.manual_seed(RNG)
np.random.seed(RNG)

# Constants
END_LABEL = "Ø"
VOWELS = set(list("aeiou"))

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
    
    test_loading = '''# Load test set
acc_df = pd.read_parquet(os.path.join(DATA_FOLDER, "cleaned_data_df.parquet"))

print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)
print(f"training: {gold_df.shape}")
print(f"test: {acc_df.shape}")
print(f"models folder: {MODELS_FOLDER}")
print("=" * 60)'''
    nb.cells.append(new_code_cell(test_loading))


def add_preprocessing_section(nb):
    """Add tokenization and feature extraction."""
    
    tokenize_code = '''pattern = re.compile("|".join(sorted(graphemes, key=len, reverse=True)))

def tokenize_morphemes(morphs):
    """Break morphemes into grapheme tokens."""
    return [pattern.findall(m.lower()) for m in morphs]

gold_df["Char_split"] = gold_df["Morph_split"].apply(tokenize_morphemes)'''
    nb.cells.append(new_code_cell(tokenize_code))
    
    cv_code = '''vowels_set = {"a", "i", "e", "o", "u"}

def grapheme_to_cv(grapheme):
    return "V" if grapheme in vowels_set else "C"

def morphs_to_cv(morphs):
    """Convert grapheme lists to CV patterns."""
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


def add_suffix_classifier_helpers(nb):
    """Add helper functions for suffix classification."""
    
    helpers = '''def safe_literal_list(obj):
    """Parse string list representation to actual list."""
    if isinstance(obj, list):
        return obj
    if pd.isna(obj):
        return None
    s = str(obj).strip()
    try:
        return ast.literal_eval(s)
    except Exception:
        return None

def flatten_char_split(char_split):
    """Flatten list-of-lists into single list."""
    if not isinstance(char_split, list):
        return None
    out = []
    for seg in char_split:
        if isinstance(seg, list):
            out.extend([str(x) for x in seg])
        else:
            out.append(str(seg))
    return out

def tokens_to_word(tokens):
    """Join tokens to surface word."""
    if not tokens:
        return ""
    return "".join(tokens)

def split_chain(chain: str):
    if chain is None:
        return []
    s = str(chain).strip()
    return [] if not s else s.split('-')

def extract_root_and_trimmed(full_chain: str):
    segs = split_chain(full_chain)
    if not segs:
        return "", END_LABEL
    root = segs[0]
    trimmed = '-'.join(segs[1:]) if len(segs) > 1 else END_LABEL
    return root, trimmed

def suffixes_from_trimmed(trimmed: str):
    if trimmed is None or trimmed == END_LABEL or str(trimmed).strip() == "":
        return []
    return str(trimmed).split('-')'''
    nb.cells.append(new_code_cell(helpers))


def add_suffix_feature_extraction(nb):
    """Add feature extraction for suffix classification."""
    
    feature_code = '''def root_cv_features(root_cv: str):
    s = root_cv or ""
    L = len(s)
    feats = {
        "root_cv": s,
        "root_len": L,
        "root_end": s[-1:] if L else "",
        "root_start": s[:1] if L else "",
        "root_suffix2": s[-2:] if L >= 2 else s,
        "root_prefix2": s[:2] if L >= 2 else s,
        "num_C": s.count('C'),
        "num_V": s.count('V'),
        "has_CC": int('CC' in s),
        "has_VV": int('VV' in s),
    }
    for i in range(L-1):
        feats[f"bg_{s[i:i+2]}"] = 1
    for i in range(L-2):
        feats[f"tg_{s[i:i+3]}"] = 1
    return feats

def last_char_features(word: str, k_chars=(1,2,3)):
    feats = {}
    if not word:
        return feats
    w = word
    for k in k_chars:
        s = w[-k:] if len(w) >= k else w
        feats[f"last{k}"] = s
    last = w[-1]
    feats["last_is_vowel"] = int(last in VOWELS)
    feats["last_char"] = last
    last_vowel = ''
    for ch in reversed(w):
        if ch in VOWELS:
            last_vowel = ch.lower()
            break
    feats["last_vowel"] = last_vowel
    return feats

def last_cluster_features(char_tokens: list, k_clusters=(1,2)):
    feats = {}
    if not char_tokens:
        return feats
    toks = char_tokens
    for k in k_clusters:
        tail = toks[-k:] if len(toks) >= k else toks
        feats[f"lastTok{k}"] = "|".join(tail)
    feats["lastTok1"] = toks[-1]
    return feats

def cv_tail_features(word: str):
    """Approximate CV tail from raw word."""
    if not word:
        return {}
    def cv(c):
        return 'V' if c in VOWELS else 'C'
    tail_cv = ''.join(cv(ch) for ch in word[-3:])
    return {"tail_cv_approx": tail_cv, "tail_last_cv": tail_cv[-1:]}

def build_features_row(row):
    """Build feature dict from a dataframe row."""
    feats = {}
    feats.update(root_cv_features(row.get("root_cv", "")))

    word = ""
    char_tokens = None

    if "Char_split" in row and row["Char_split"] is not None:
        cs = safe_literal_list(row["Char_split"])
        toks = flatten_char_split(cs) if cs is not None else None
        char_tokens = toks
        word = tokens_to_word(toks) if toks else ""
    elif "Word" in row and pd.notna(row.get("Word", None)):
        word = str(row["Word"])
    else:
        word = ""

    feats.update(last_char_features(word, k_chars=(1,2,3)))
    if char_tokens:
        feats.update(last_cluster_features(char_tokens, k_clusters=(1,2)))
    feats.update(cv_tail_features(word))

    # Numeric features
    for k in NEW_NUM_FEATS:
        if k in row and pd.notna(row[k]):
            try:
                feats[k] = float(row[k])
            except Exception:
                pass

    return feats'''
    nb.cells.append(new_code_cell(feature_code))


def add_suffix_classifier_functions(nb):
    """Add suffix classifier training functions."""
    
    build_dataset = '''def build_dataset(df_in: pd.DataFrame):
    rows = []
    for _, r in df_in.iterrows():
        full = r['Full_chain']
        root, trimmed_auto = extract_root_and_trimmed(full)
        trimmed = r['Trimmed_chain'] if 'Trimmed_chain' in df_in.columns and pd.notna(r['Trimmed_chain']) else trimmed_auto
        suffixes = suffixes_from_trimmed(trimmed)

        row = {
            "full_chain": full,
            "root_cv": root,
            "trimmed": trimmed if trimmed else END_LABEL,
            "suffixes": suffixes,
            "suffix_len": len(suffixes),
        }
        for opt in ("Word", "Char_split", "CV_split"):
            if opt in df_in.columns:
                row[opt] = r[opt]

        for k in NEW_NUM_FEATS:
            if k in df_in.columns:
                row[k] = r[k]

        rows.append(row)
    return pd.DataFrame(rows)

def dicts_from_df(df: pd.DataFrame, add_prev=None):
    """Turn rows into feature dicts."""
    feat_dicts = []
    for _, r in df.iterrows():
        base = build_features_row(r)
        if add_prev:
            for k in add_prev:
                if k in r and pd.notna(r[k]):
                    base[k] = r[k]
        feat_dicts.append(base)
    return feat_dicts

def vec_fit_transform(feat_dicts):
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(feat_dicts)
    return X, vec

def vec_transform(vec, feat_dicts):
    return vec.transform(feat_dicts)

def grouped_split(df, train_size=0.8, seed=RANDOM_STATE):
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    groups = df['root_cv'].astype(str).values
    tr_idx, te_idx = next(gss.split(df, groups=groups))
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)'''
    nb.cells.append(new_code_cell(build_dataset))
    
    metrics_code = '''def topN_labels_by_freq(y, Ns=(16,25,37,57,103)):
    ctr = Counter(y)
    most_common = ctr.most_common()
    return {N: set([lab for lab,_ in most_common[:N]]) for N in Ns}, ctr

def eval_subsets(y_true, y_pred, labels_by_topN):
    out = {}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    for N, labelset in labels_by_topN.items():
        idx = [i for i,lab in enumerate(y_true) if lab in labelset]
        if not idx:
            out[N] = {"accuracy": np.nan, "f1_macro": np.nan, "f1_weighted": np.nan, "support": 0}
            continue
        yt, yp = y_true[idx], y_pred[idx]
        out[N] = {
            "accuracy": accuracy_score(yt, yp),
            "f1_macro": f1_score(yt, yp, average='macro', zero_division=0),
            "f1_weighted": f1_score(yt, yp, average='weighted', zero_division=0),
            "support": len(idx),
        }
    return out

def print_subset_metrics(name, d):
    print(f"\\n== {name}: Top-N subsets ==")
    for N in sorted(d.keys()):
        m = d[N]
        print(f"Top-{N:>3} (n={m['support']:>4}): Acc={m['accuracy']:.3f} | F1_mac={m['f1_macro']:.3f} | F1_wt={m['f1_weighted']:.3f}")'''
    nb.cells.append(new_code_cell(metrics_code))
    
    classifiers = '''def make_classifier(kind="tree", **kwargs):
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=80, max_depth=10, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1
        )
    return DecisionTreeClassifier(
        criterion="entropy",
        max_depth=kwargs.get("max_depth", 6),
        min_samples_leaf=kwargs.get("min_samples_leaf", 10),
        random_state=RANDOM_STATE
    )

def run_single_shot(df_all, clf_kind="tree"):
    df_tr, df_te = grouped_split(df_all, train_size=0.8)
    Xtr_dicts = dicts_from_df(df_tr)
    Xtr, vec = vec_fit_transform(Xtr_dicts)
    ytr = df_tr['trimmed'].astype(str).values

    clf = make_classifier(clf_kind)
    clf.fit(Xtr, ytr)

    Xte = vec_transform(vec, dicts_from_df(df_te))
    yte = df_te['trimmed'].astype(str).values
    yhat = clf.predict(Xte)

    acc = accuracy_score(yte, yhat)
    f1m = f1_score(yte, yhat, average='macro', zero_division=0)
    f1w = f1_score(yte, yhat, average='weighted', zero_division=0)

    print("=== single-shot classifier ===")
    print(f"test: acc={acc:.3f} | F1_macro={f1m:.3f} | F1_weighted={f1w:.3f}")

    labels_by_topN, _ = topN_labels_by_freq(df_tr['trimmed'].astype(str).values)
    subset = eval_subsets(yte, yhat, labels_by_topN)
    print_subset_metrics("single-shot", subset)

    if ("Word" not in df_all.columns) and ("Char_split" not in df_all.columns):
        print("\\n[warn] no Word/Char_split columns - using root-only features")

    return {"clf": clf, "vec": vec, "test_df": df_te, "test_pred": yhat}'''
    nb.cells.append(new_code_cell(classifiers))


def add_sequential_classifier(nb):
    """Add sequential suffix classifier."""
    
    seq_code = '''def train_length_classifier(df_tr, clf_kind="tree"):
    ylen = []
    for n in df_tr['suffix_len'].values:
        ylen.append(str(n) if n in (1,2,3) else "4+")
    ylen = np.array(ylen)

    X_dicts = dicts_from_df(df_tr)
    X, vec = vec_fit_transform(X_dicts)
    clf = make_classifier(clf_kind, max_depth=5, min_samples_leaf=10)
    clf.fit(X, ylen)
    return clf, vec

def make_step_frame(df, step):
    y = []
    for sufs in df['suffixes']:
        if len(sufs) >= step:
            y.append(sufs[-step])
        else:
            y.append(END_LABEL)
    df2 = df.copy()
    df2[f"y_step{step}"] = y
    return df2

def run_sequential(df_all, clf_kind="tree", max_steps_cap=5):
    df_tr, df_te = grouped_split(df_all, train_size=0.8)

    len_clf, len_vec = train_length_classifier(df_tr, clf_kind=clf_kind)

    max_steps = min(max_steps_cap, 4)
    print(f"\\n=== length-first + sequential ===")
    print(f"training up to {max_steps} steps (last→first)")

    step_vecs, step_clfs = {}, {}
    prev_cols = []
    for step in range(1, max_steps+1):
        df_step = make_step_frame(df_tr, step)
        X_dicts = dicts_from_df(df_step, add_prev=set(prev_cols))
        X, vec = vec_fit_transform(X_dicts)
        y = df_step[f"y_step{step}"].astype(str).values

        clf = make_classifier(clf_kind, max_depth=6, min_samples_leaf=8)
        clf.fit(X, y)

        step_vecs[step] = vec
        step_clfs[step] = clf
        prev_cols.append(f"y_step{step}")

    gold_full = df_te['trimmed'].astype(str).values
    preds_full = []

    per_step_gold = defaultdict(list)
    per_step_pred = defaultdict(list)

    Xlen = vec_transform(len_vec, dicts_from_df(df_te))
    ylen_pred = len_clf.predict(Xlen)

    for i, r in df_te.iterrows():
        k_str = ylen_pred[i]
        K = 4 if k_str == "4+" else int(k_str)

        prev_preds = []
        base_row = r.to_dict()

        for step in range(1, K+1):
            feat = build_features_row(base_row)
            for j, lab in enumerate(prev_preds, start=1):
                feat[f"y_step{j}"] = lab

            X_one = vec_transform(step_vecs[step], [feat])
            yhat = step_clfs[step].predict(X_one)[0]

            gold_suffixes = r['suffixes']
            ygold = gold_suffixes[-step] if len(gold_suffixes) >= step else END_LABEL
            per_step_gold[step].append(ygold)
            per_step_pred[step].append(yhat)

            if yhat == END_LABEL:
                break
            prev_preds.append(yhat)

        pred_chain = '-'.join(reversed(prev_preds)) if prev_preds else END_LABEL
        preds_full.append(pred_chain)

    acc = accuracy_score(gold_full, preds_full)
    f1m = f1_score(gold_full, preds_full, average='macro', zero_division=0)
    f1w = f1_score(gold_full, preds_full, average='weighted', zero_division=0)

    print(f"test: acc={acc:.3f} | F1_macro={f1m:.3f} | F1_weighted={f1w:.3f}")

    labels_by_topN, _ = topN_labels_by_freq(df_tr['trimmed'].astype(str).values)
    subset = eval_subsets(gold_full, preds_full, labels_by_topN)
    print_subset_metrics("sequential", subset)

    for step in range(1, max_steps+1):
        if len(per_step_gold[step]) == 0:
            continue
        ys = np.array(per_step_gold[step])
        ps = np.array(per_step_pred[step])
        a = accuracy_score(ys, ps)
        fm = f1_score(ys, ps, average='macro', zero_division=0)
        fw = f1_score(ys, ps, average='weighted', zero_division=0)
        print(f"step {step}: acc={a:.3f} | F1_macro={fm:.3f} | F1_weighted={fw:.3f}")

    if ("Word" not in df_all.columns) and ("Char_split" not in df_all.columns):
        print("\\n[warn] no Word/Char_split found")

    return {"len_clf": len_clf, "len_vec": len_vec,
            "step_clfs": step_clfs, "step_vecs": step_vecs,
            "test_df": df_te, "test_pred": preds_full}'''
    nb.cells.append(new_code_cell(seq_code))


def add_suffix_classifier_io(nb):
    """Add save/load functions for suffix classifiers."""
    
    io_code = '''def generate_suffix_classifier_id(str_df, clf_kind="tree"):
    """Hash data/params to get unique classifier ID."""
    params_dict = {
        'clf_kind': clf_kind,
        'df_shape': str_df.shape if str_df is not None else (0, 0),
        'df_columns': sorted(str_df.columns.tolist()) if str_df is not None else []
    }
    params_str = json.dumps(params_dict, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()[:16]

def save_suffix_classifiers(single, seq, classifier_id, data_folder=DATA_FOLDER):
    """Save suffix classifiers."""
    classifier_dir = os.path.join(data_folder, f"suffix_classifiers_{classifier_id}")
    os.makedirs(classifier_dir, exist_ok=True)
    
    if single is not None:
        with open(os.path.join(classifier_dir, "single.pkl"), "wb") as f:
            pickle.dump(single, f)
    
    if seq is not None:
        with open(os.path.join(classifier_dir, "seq.pkl"), "wb") as f:
            pickle.dump(seq, f)
    
    with open(os.path.join(classifier_dir, "metadata.json"), "w") as f:
        json.dump({
            'classifier_id': classifier_id,
            'clf_kind': single.get('clf').__class__.__name__ if single and 'clf' in single else 'unknown'
        }, f, indent=2)
    
    print(f"saved suffix classifiers to {classifier_dir}")
    return classifier_dir

def load_suffix_classifiers(classifier_id, data_folder=DATA_FOLDER):
    """Load suffix classifiers."""
    classifier_dir = os.path.join(data_folder, f"suffix_classifiers_{classifier_id}")
    
    if not os.path.exists(classifier_dir):
        return None, None
    
    single = None
    seq = None
    
    single_path = os.path.join(classifier_dir, "single.pkl")
    seq_path = os.path.join(classifier_dir, "seq.pkl")
    
    if os.path.exists(single_path):
        with open(single_path, "rb") as f:
            single = pickle.load(f)
    
    if os.path.exists(seq_path):
        with open(seq_path, "rb") as f:
            seq = pickle.load(f)
    
    if single is not None or seq is not None:
        print(f"loaded suffix classifiers from {classifier_dir}")
    
    return single, seq

def run_all(str_df, clf_kind="tree"):
    """Train or load suffix classifiers."""
    classifier_id = generate_suffix_classifier_id(str_df, clf_kind=clf_kind)
    
    print(f"looking for suffix classifiers {classifier_id}...")
    single, seq = load_suffix_classifiers(classifier_id, data_folder=DATA_FOLDER)
    
    if single is not None and seq is not None:
        print(f"found them! skipping training")
        return single, seq
    
    print(f"not found, training...")
    
    df_all = build_dataset(str_df)
    print(f"samples: {len(df_all)}")
    print(f"unique trimmed: {df_all['trimmed'].nunique()}")
    print(f"suffix len dist: {df_all['suffix_len'].value_counts().sort_index().to_dict()}")

    single = run_single_shot(df_all, clf_kind=clf_kind)
    seq = run_sequential(df_all, clf_kind=clf_kind, max_steps_cap=5)
    
    print(f"\\nsaving classifiers {classifier_id}...")
    save_suffix_classifiers(single, seq, classifier_id, data_folder=DATA_FOLDER)
    
    return single, seq'''
    nb.cells.append(new_code_cell(io_code))


def add_run_suffix_classifiers(nb):
    """Add cells to run suffix classifiers."""
    
    run_code = '''single, seq = run_all(str_df, clf_kind="tree")'''
    nb.cells.append(new_code_cell(run_code))
    
    rf_code = '''single, seq = run_all(str_df, clf_kind="rf")'''
    nb.cells.append(new_code_cell(rf_code))


def add_boundary_helpers(nb):
    """Add helper functions for boundary prediction."""
    
    helpers = '''def safe_list(x):
    """Handle various list formats from dataframes."""
    if isinstance(x, list):
        return x
    s = str(x)
    try:
        return ast.literal_eval(s)
    except Exception:
        s2 = s.replace("[[", "[['").replace("]]", "']]").replace("], [", "'],['").replace(", ", "','")
        return ast.literal_eval(s2)

def flatten(list_of_lists):
    """Flatten nested list."""
    out = []
    for seg in list_of_lists:
        out.extend(seg)
    return [str(t) for t in out]

def extract_priv_features_from_row(row, feat_names):
    """Pull numeric features from row into vector."""
    vec = []
    for k in feat_names:
        val = row[k] if (k in row and pd.notna(row[k])) else 0.0
        try:
            vec.append(float(val))
        except Exception:
            vec.append(0.0)
    return vec

def build_samples_with_priv(df, feat_names=NEW_NUM_FEATS):
    """Convert dataframe rows to sample dicts for training."""
    rows = []
    for _, r in df.iterrows():
        cs = safe_list(r["Char_split"])
        toks = flatten(cs)
        lens = [len(seg) for seg in cs]
        cut_idxs = set(np.cumsum(lens)[:-1].tolist())
        y = [1 if (i + 1) in cut_idxs else 0 for i in range(len(toks) - 1)]
        priv = extract_priv_features_from_row(r, feat_names)
        rows.append({"tokens": toks, "y": y, "priv": priv})
    return rows'''
    nb.cells.append(new_code_cell(helpers))


def add_dt_prior_section(nb):
    """Add Decision Tree prior functions."""
    
    dt_code = '''def featurize_window(tokens, i, k_left=2, k_right=2):
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

def train_dt_prior(samples, max_depth=6, min_leaf=8):
    """Train DT to predict boundary probabilities from local context."""
    Xdict, y = [], []
    for s in samples:
        T = len(s["tokens"])
        for i in range(T - 1):
            Xdict.append(featurize_window(s["tokens"], i))
            y.append(s["y"][i])
    
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(Xdict)
    
    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        random_state=RNG
    )
    clf.fit(X, y)
    
    print(f"DT prior: {clf.tree_.node_count} nodes, depth={clf.tree_.max_depth}")
    return clf, vec

def prior_probs_for_sample(clf, vec, tokens):
    """Get boundary probabilities from DT for a tokenized word."""
    if clf is None or vec is None or len(tokens) <= 1:
        return [0.5] * (max(len(tokens) - 1, 0))
    
    Xd = [featurize_window(tokens, i) for i in range(len(tokens) - 1)]
    X = vec.transform(Xd)
    proba = clf.predict_proba(X)
    return proba[:, 1].tolist()'''
    nb.cells.append(new_code_cell(dt_code))


def add_k_teacher_section(nb):
    """Add K-teacher regularization code."""
    
    k_teacher = '''def train_k_teacher_priv(samples, feat_dim):
    """Train regressor to predict number of cuts from features."""
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
    
    dataset_code = '''def build_vocab(samples, min_freq=1):
    """Build token vocabulary from samples."""
    ctr = Counter()
    for s in samples:
        ctr.update(s["tokens"])
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for t, c in sorted(ctr.items(), key=lambda x: (-x[1], x[0])):
        if c >= min_freq and t not in vocab:
            vocab[t] = len(vocab)
    return vocab

class SegDataset(Dataset):
    """Dataset for boundary prediction training."""
    def __init__(self, samples, vocab, dt_clf=None, dt_vec=None, feat_dim=0):
        self.samples = samples
        self.vocab = vocab
        self.dt_clf = dt_clf
        self.dt_vec = dt_vec
        self.feat_dim = feat_dim

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        tokens = s["tokens"]
        ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]
        y = s["y"]
        prior = prior_probs_for_sample(self.dt_clf, self.dt_vec, tokens)
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
    Can fuse DT prior via concatenation or logit addition.
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
    """Compute P/R/F1 for boundary prediction."""
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

        loss = criterion_ce(logits_flat, y_true)

        if lambda_prior > 0:
            cut_logit = logits[..., 1]
            prior_flat = prior[mask_b]
            loss_pr = criterion_bce(cut_logit[mask_b], prior_flat)
            loss = loss + lambda_prior * loss_pr

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
    
    io_code = '''def generate_model_id(df, epochs, use_prior, fuse_mode, lambda_prior, lambda_k,
                      batch_size, hparams, max_depth=6, min_leaf=8):
    """Hash training params to get unique model ID."""
    params_dict = {
        'epochs': epochs,
        'use_prior': use_prior,
        'fuse_mode': fuse_mode,
        'lambda_prior': lambda_prior,
        'lambda_k': lambda_k,
        'batch_size': batch_size,
        'hparams': hparams,
        'max_depth': max_depth,
        'min_leaf': min_leaf,
        'df_shape': df.shape if df is not None else (0, 0)
    }
    params_str = json.dumps(params_dict, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()[:16]

def save_dt_prior(dt_clf, dt_vec, model_id, models_folder=MODELS_FOLDER):
    """Save DT prior."""
    model_dir = os.path.join(models_folder, model_id)
    os.makedirs(model_dir, exist_ok=True)
    
    with open(os.path.join(model_dir, "dt_clf.pkl"), "wb") as f:
        pickle.dump(dt_clf, f)
    with open(os.path.join(model_dir, "dt_vec.pkl"), "wb") as f:
        pickle.dump(dt_vec, f)
    print(f"saved DT prior to {model_dir}")

def load_dt_prior(model_id, models_folder=MODELS_FOLDER):
    """Load DT prior."""
    model_dir = os.path.join(models_folder, model_id)
    dt_clf_path = os.path.join(model_dir, "dt_clf.pkl")
    dt_vec_path = os.path.join(model_dir, "dt_vec.pkl")
    
    if not os.path.exists(dt_clf_path) or not os.path.exists(dt_vec_path):
        return None, None
    
    with open(dt_clf_path, "rb") as f:
        dt_clf = pickle.load(f)
    with open(dt_vec_path, "rb") as f:
        dt_vec = pickle.load(f)
    print(f"loaded DT prior from {model_dir}")
    return dt_clf, dt_vec

def save_model(model, vocab, out, model_id, models_folder=MODELS_FOLDER):
    """Save model weights and artifacts."""
    model_dir = os.path.join(models_folder, model_id)
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
    
    with open(os.path.join(model_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    
    if out.get("dt_clf") is not None and out.get("dt_vec") is not None:
        save_dt_prior(out["dt_clf"], out["dt_vec"], model_id, models_folder)
    
    artifacts = {k: v for k, v in out.items() if k not in ["dt_clf", "dt_vec"]}
    with open(os.path.join(model_dir, "artifacts.pkl"), "wb") as f:
        pickle.dump(artifacts, f)
    
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump({
            'model_id': model_id,
            'vocab_size': len(vocab),
            'model_name': MODEL_NAME
        }, f, indent=2)
    
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
    
    dt_clf, dt_vec = load_dt_prior(model_id, models_folder)
    
    artifacts_path = os.path.join(model_dir, "artifacts.pkl")
    if not os.path.exists(artifacts_path):
        return None
    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)
    
    out = {**artifacts, "dt_clf": dt_clf, "dt_vec": dt_vec}
    
    model_path = os.path.join(model_dir, "model.pt")
    if not os.path.exists(model_path):
        return None
    
    print(f"loaded artifacts from {model_dir}")
    return {
        'vocab': vocab,
        'out': out,
        'dt_clf': dt_clf,
        'dt_vec': dt_vec,
        'model_state_path': model_path,
        'model_dir': model_dir
    }'''
    nb.cells.append(new_code_cell(io_code))


def add_main_training_function(nb):
    """Add the main training orchestration function."""
    
    main_train = '''def run_segmentation_with_privK(
    df,
    epochs=15,
    use_prior=True,
    fuse_mode="logit_add",
    lambda_prior=0.1,
    lambda_k=0.2,
    batch_size=64,
    hparams=None,
    max_depth=6,
    min_leaf=8
):
    """Train or load a segmentation model with DT priors."""
    if hparams is None:
        hparams = dict(emb_dim=16, hidden_size=64, num_layers=2,
                       dropout=0.25, lr=1e-3, weight_decay=1e-4, freeze_emb=False)
    
    model_id = generate_model_id(
        df, epochs, use_prior, fuse_mode, lambda_prior, lambda_k,
        batch_size, hparams, max_depth=max_depth, min_leaf=min_leaf
    )
    
    print(f"looking for model {model_id}...")
    loaded = load_model(model_id, models_folder=MODELS_FOLDER)
    
    if loaded is not None:
        print(f"found it! loading from {loaded['model_dir']}")
        vocab = loaded['vocab']
        out = loaded['out']
        dt_clf = loaded['dt_clf']
        dt_vec = loaded['dt_vec']
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
    
    print(f"not found, training from scratch...")
    
    samples = build_samples_with_priv(df, feat_names=NEW_NUM_FEATS)
    train_s, test_s = split_train_test(samples, 0.2)

    dt_clf, dt_vec = (None, None)
    if use_prior:
        dt_clf, dt_vec = load_dt_prior(model_id, models_folder=MODELS_FOLDER)
        if dt_clf is None or dt_vec is None:
            print("training DT prior...")
            dt_clf, dt_vec = train_dt_prior(train_s, max_depth=max_depth, min_leaf=min_leaf)
        else:
            print("using existing DT prior")

    feat_dim = len(NEW_NUM_FEATS)
    k_reg = train_k_teacher_priv(train_s, feat_dim=feat_dim)

    vocab = build_vocab(train_s, min_freq=1)

    train_ds = SegDataset(train_s, vocab, dt_clf, dt_vec, feat_dim=feat_dim)
    test_ds = SegDataset(test_s, vocab, dt_clf, dt_vec, feat_dim=feat_dim)
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
        "dt_clf": dt_clf,
        "dt_vec": dt_vec,
        "k_teacher": k_reg,
        "best_thr": best_thr
    }
    
    print(f"saving model {model_id}...")
    save_model(model, vocab, out, model_id, models_folder=MODELS_FOLDER)

    return model, vocab, out'''
    nb.cells.append(new_code_cell(main_train))


def add_kfold_section(nb):
    """Add k-fold cross-validation function."""
    
    kfold_code = '''def run_kfold_cross_validation(
    df,
    n_folds=5,
    epochs=15,
    use_prior=True,
    fuse_mode="logit_add",
    lambda_prior=0.1,
    lambda_k=0.2,
    batch_size=64,
    hparams=None,
    max_depth=6,
    min_leaf=8,
    random_state=42
):
    """K-fold cross-validation for more robust evaluation."""
    if hparams is None:
        hparams = dict(emb_dim=16, hidden_size=64, num_layers=2,
                       dropout=0.25, lr=1e-3, weight_decay=1e-4, freeze_emb=False)
    
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
        
        dt_clf, dt_vec = (None, None)
        if use_prior:
            print("training DT prior...")
            dt_clf, dt_vec = train_dt_prior(train_samples, max_depth=max_depth, min_leaf=min_leaf)
        
        feat_dim = len(NEW_NUM_FEATS)
        k_reg = train_k_teacher_priv(train_samples, feat_dim=feat_dim)
        vocab = build_vocab(train_samples, min_freq=1)
        
        train_ds = SegDataset(train_samples, vocab, dt_clf=dt_clf, dt_vec=dt_vec, feat_dim=feat_dim)
        val_ds = SegDataset(val_samples, vocab, dt_clf=dt_clf, dt_vec=dt_vec, feat_dim=feat_dim)
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
def segment_tokens(model, vocab, tokens, dt_clf=None, dt_vec=None, thr=0.5):
    """Segment a tokenized word and return the segmented string + probabilities."""
    ids = torch.tensor([[vocab.get(t, vocab["<UNK>"]) for t in tokens]], dtype=torch.long)
    mask_tok = torch.ones_like(ids, dtype=torch.bool)
    T = len(tokens)
    if T <= 1:
        return "".join(tokens), np.array([])
    
    prior_list = prior_probs_for_sample(dt_clf, dt_vec, tokens)
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
    dt_clf, dt_vec = out["dt_clf"], out["dt_vec"]
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
        seg_string, probs = segment_tokens(model, vocab, toks, dt_clf=dt_clf, dt_vec=dt_vec, thr=thr)
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
    dt_clf, dt_vec = out["dt_clf"], out["dt_vec"]
    thr = float(out.get("best_thr", 0.5)) if use_tuned_thr else 0.5
    allowed_suffixes_set = set(allowed_suffixes)

    total_tp = total_fp = total_fn = 0
    exact_hits = 0
    n_total_words = 0
    n_evaluated_words = 0
    rejection_count = 0
    examples = []

    for _, row in df.iterrows():
        word = str(row["Word"])
        gold_variants = normalize_gold_variants(row["Gold"])

        if not isinstance(gold_variants, list) or len(gold_variants) == 0:
            continue
        
        n_total_words += 1

        toks = tokenize_with_vocab(word, vocab, max_token_len=max_token_len)
        seg_string, probs = segment_tokens(model, vocab, toks, dt_clf=dt_clf, dt_vec=dt_vec, thr=thr)
        predicted_morphs = seg_string.split('-')

        if not is_segmentation_valid(predicted_morphs, allowed_suffixes_set):
            rejection_count += 1
            continue

        n_evaluated_words += 1
        
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

    print(f"tried {n_total_words} words")
    print(f"rejected {rejection_count} ({rejection_count/n_total_words:.1%}) with invalid suffixes")
    print(f"scoring {n_evaluated_words} valid predictions")
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
    
    return {"micro_f1": micro_F1, "exact_match_rate": exact_rate, "rejection_count": rejection_count}'''
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

filename = os.path.join(DATA_FOLDER, "suffixesCQ-Anettte-Rios_LS.txt")
suffix_list = read_suffixes(filename)
print(f"loaded {len(suffix_list)} suffixes")'''
    nb.cells.append(new_code_cell(suffix_load))


def add_best_hparams_section(nb):
    """Add best hyperparameters."""
    
    best_hparams = '''# Hyperparameters
best = {
    "emb_dim": 16,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.25,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "freeze_emb": False,
}'''
    nb.cells.append(new_code_cell(best_hparams))


def add_train_model_section(nb):
    """Add model training cell."""
    
    train_model = '''# Train or load the model
model, vocab, out = run_segmentation_with_privK(
    str_df,
    epochs=50,
    use_prior=True,
    fuse_mode="logit_add",
    lambda_prior=0.1,
    lambda_k=0.2,
    batch_size=64,
    hparams=best
)'''
    nb.cells.append(new_code_cell(train_model))


def add_run_kfold_section(nb):
    """Add k-fold CV execution."""
    
    run_kfold = '''# Run k-fold cross-validation
kfold_results = run_kfold_cross_validation(
    df=str_df,
    n_folds=5,
    epochs=15,
    use_prior=True,
    fuse_mode="logit_add",
    lambda_prior=0.1,
    lambda_k=0.2,
    hparams=best,
    max_depth=6,
    min_leaf=8,
    random_state=RNG
)

print(f"\\navg exact match: {kfold_results['mean_metrics']['exact_match']:.3f} +/- {kfold_results['std_metrics']['exact_match']:.3f}")
print(f"avg boundary F1: {kfold_results['mean_metrics']['boundary_f1']:.3f} +/- {kfold_results['std_metrics']['boundary_f1']:.3f}")'''
    nb.cells.append(new_code_cell(run_kfold))


def add_example_usage_section(nb):
    """Add example word segmentation."""
    
    example = '''# Example segmentation
word = "pikunas"
tokens = tokenize_with_vocab(word, vocab, max_token_len=4)
thr = out.get("best_thr", 0.5)

seg_string, boundary_probs = segment_tokens(
    model, vocab, tokens, dt_clf=out["dt_clf"], dt_vec=out["dt_vec"], thr=thr
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
    add_suffix_classifier_helpers(nb)
    add_suffix_feature_extraction(nb)
    add_suffix_classifier_functions(nb)
    add_sequential_classifier(nb)
    add_suffix_classifier_io(nb)
    add_run_suffix_classifiers(nb)
    add_boundary_helpers(nb)
    add_dt_prior_section(nb)
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
    add_best_hparams_section(nb)
    add_train_model_section(nb)
    add_run_kfold_section(nb)
    add_example_usage_section(nb)
    add_final_eval_section(nb)
    
    # Save notebook
    output_path = "DT-LSTM-MarkovFilter_refactored.ipynb"
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print(f"Refactored notebook saved to: {output_path}")


if __name__ == "__main__":
    main()

