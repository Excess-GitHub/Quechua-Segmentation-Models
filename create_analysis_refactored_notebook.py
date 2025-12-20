#!/usr/bin/env python3
"""
Script to generate a refactored version of the analysis notebook.
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
        "# Statistical Analysis of Quechua Morphology and Corpus\n\n"
        "Analyzes Zipf's law, Heaps' law, and Zipf-Mandelbrot fitting on Quechua data. "
        "Helps understand morpheme distributions and vocabulary growth patterns."
    ))


def add_imports_section(nb):
    """Add all imports in a single organized cell."""
    imports_code = '''# Core libraries
import os
import ast
import math
import re
from collections import Counter

# Data handling
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt'''
    nb.cells.append(new_code_cell(imports_code))


def add_config_section(nb):
    """Add configuration constants."""
    config_code = '''# Paths
DATA_FOLDER = "data"
FILE_PATH = os.path.join(DATA_FOLDER, "qu_merged_dump.txt")

# Tokenization options
lowercase = True
keep_apostrophes = True

# Analysis parameters
tail_ignore = 50
tail_min_len = 100
heaps_stride = 50
min_rank_for_fit = 1
max_rank_for_fit = None'''
    nb.cells.append(new_code_cell(config_code))


def add_zipf_morphemes_section(nb):
    """Add Part 1: Zipf's law analysis on morphological tokens."""
    
    zipf_morphemes = '''# Part 1: Zipf's law on morphological tokens

# Load gold standard data
df = pd.read_parquet(os.path.join(DATA_FOLDER, "Sue_kalt.parquet"))
df['Word'] = df['word']
df['morph'] = df['morph'].str.replace('-', ' ')
df['Morph_split_str'] = df['morph']
df['Morph_split'] = df['morph'].str.split(' ')
df = df[['Word', 'Morph_split', 'Morph_split_str']]

# Extract all morpheme tokens
tokens = []
for toks in df["Morph_split"]:
    tokens.extend(t for t in toks if isinstance(t, str) and t.strip() != "")

# Count frequencies
freq = Counter(tokens)
counts = np.array(sorted(freq.values(), reverse=True), dtype=np.int64)

# Zipf plot for morphemes
ranks = np.arange(1, len(counts) + 1, dtype=np.int64)

plt.figure(figsize=(7, 5))
plt.loglog(ranks, counts, marker='o', linestyle='none', markersize=3)
plt.xlabel("Rank (log)")
plt.ylabel("Frequency (log)")
plt.title("Zipf plot for morph tokens (Morph_split)")

# Fit line on tail to estimate exponent
k = min(50, len(counts) // 10 if len(counts) > 1000 else 10)
if len(counts) > k + 10:
    x = np.log(ranks[k:])
    y = np.log(counts[k:])
    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept
    plt.loglog(ranks[k:], np.exp(y_fit), linewidth=1)

    s_est = -slope
    print(f"estimated zipf exponent s (tail fit): {s_est:.3f}")

plt.tight_layout()
plt.show()'''
    nb.cells.append(new_code_cell(zipf_morphemes))


def add_corpus_analysis_section(nb):
    """Add Part 2: Corpus-level statistical analysis."""
    
    corpus_analysis = '''# Part 2: Corpus-level analysis

# Tokenizer
if keep_apostrophes:
    TOKEN_RE = re.compile(r"[^\\W\\d_]+(?:[''][^\\W\\d_]+)?", flags=re.UNICODE)
else:
    TOKEN_RE = re.compile(r"[^\\W\\d_]+", flags=re.UNICODE)

def iter_tokens_from_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if lowercase:
                line = line.lower()
            for m in TOKEN_RE.finditer(line):
                yield m.group(0)

# Count tokens in corpus
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"file not found: {FILE_PATH}")

freq = Counter()
for tok in iter_tokens_from_file(FILE_PATH):
    freq[tok] += 1

total_tokens = sum(freq.values())
vocab_size = len(freq)

print(f"file: {FILE_PATH}")
print(f"total tokens: {total_tokens:,}")
print(f"vocabulary size: {vocab_size:,}")

print("\\ntop 25 tokens:")
for i, (tok, c) in enumerate(freq.most_common(25), 1):
    print(f"{i:>2}. {tok}\\t{c}")

# Prepare for Zipf plot
counts = np.array(sorted(freq.values(), reverse=True), dtype=np.int64)
ranks = np.arange(1, len(counts) + 1, dtype=np.int64)

# Zipf plot for corpus words
plt.figure(figsize=(7,5))
plt.loglog(ranks, counts, marker='o', linestyle='none', markersize=3)
plt.xlabel("Rank (log)")
plt.ylabel("Frequency (log)")
plt.title("Zipf plot — " + os.path.basename(FILE_PATH))
plt.show()'''
    nb.cells.append(new_code_cell(corpus_analysis))


def add_advanced_modeling_section(nb):
    """Add Part 3: Advanced statistical modeling (Heaps' law, Zipf-Mandelbrot)."""
    
    advanced_functions = '''# Part 3: Advanced statistical modeling

# Tokenizer (same as before)
if keep_apostrophes:
    TOKEN_RE = re.compile(r"[^\\W\\d_]+(?:[''][^\\W\\d_]+)?", flags=re.UNICODE)
else:
    TOKEN_RE = re.compile(r"[^\\W\\d_]+", flags=re.UNICODE)

def iter_tokens(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if lowercase:
                line = line.lower()
            for m in TOKEN_RE.finditer(line):
                yield m.group(0)

# Heaps' law: vocabulary growth
def vocab_growth(path, stride=50):
    """Track vocabulary growth as we process tokens."""
    seen = set()
    n_points, V_points = [], []
    n = 0
    for tok in iter_tokens(path):
        n += 1
        if tok not in seen:
            seen.add(tok)
        if (n % stride) == 0:
            n_points.append(n)
            V_points.append(len(seen))
    if not n_points or n_points[-1] != n:
        n_points.append(n)
        V_points.append(len(seen))
    return np.array(n_points, dtype=np.int64), np.array(V_points, dtype=np.int64)

def fit_heaps_logls(n, V, min_n=1000, tail_frac=0.7):
    """Fit Heaps' law: V(n) = K * n^β in log-space."""
    mask = n >= max(1, min_n)
    n_fit, V_fit = (n[mask], V[mask]) if mask.any() else (n, V)

    if 0 < tail_frac < 1.0:
        start = int((1 - tail_frac) * len(n_fit))
        n_fit = n_fit[start:]
        V_fit = V_fit[start:]

    x = np.log(n_fit)
    y = np.log(V_fit)
    slope, intercept = np.polyfit(x, y, 1)
    beta = slope
    K = np.exp(intercept)
    
    yhat = slope * x + intercept
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return beta, K, r2, (n_fit[0], n_fit[-1])

# Zipf-Mandelbrot fitting
def counts_from_file(path):
    """Count word frequencies and return sorted counts."""
    freq = Counter()
    for tok in iter_tokens(path):
        freq[tok] += 1
    counts = np.array(sorted(freq.values(), reverse=True), dtype=np.int64)
    return counts, freq

def fit_zipf_mandelbrot(counts, rmin=1, rmax=None):
    """Fit Zipf-Mandelbrot model: frequency(r) = C / (r + q)^s"""
    R = np.arange(1, len(counts) + 1, dtype=np.float64)
    if rmax is None or rmax > len(R): rmax = len(R)
    r_slice = slice(rmin-1, rmax)
    r = R[r_slice]
    c = counts[r_slice].astype(np.float64)

    log_r = np.log(r)

    def rmse_for(s, q):
        rq = r + q
        if np.any(rq <= 0):
            return np.inf, None
        log_rq = np.log(rq)
        logC = np.mean(np.log(c) + s * log_rq)
        logc_hat = logC - s * log_rq
        rmse = np.sqrt(np.mean((np.log(c) - logc_hat)**2))
        return rmse, logC

    # Coarse grid search
    s_grid = np.linspace(0.6, 1.6, 27)
    q_grid = np.concatenate([np.linspace(0.0, 20.0, 21),
                             np.linspace(25.0, 200.0, 8)])
    best = (np.inf, None, None, None)
    for s in s_grid:
        for q in q_grid:
            rmse, logC = rmse_for(s, q)
            if rmse < best[0]:
                best = (rmse, s, q, logC)

    rmse0, s0, q0, logC0 = best

    # Refined local search
    def refine(s_c, q_c, s_step=0.05, q_step=2.0, n_iter=6):
        best_rmse, best_s, best_q, best_logC = rmse0, s0, q0, logC0
        for _ in range(n_iter):
            improved = False
            for s in np.linspace(best_s - s_step, best_s + s_step, 7):
                for q in np.linspace(max(-0.9, best_q - q_step), best_q + q_step, 7):
                    rmse, logC = rmse_for(s, q)
                    if rmse < best_rmse:
                        best_rmse, best_s, best_q, best_logC = rmse, s, q, logC
                        improved = True
            s_step *= 0.5
            q_step *= 0.5
            if not improved:
                break
        return best_rmse, best_s, best_q, best_logC

    rmse, s, q, logC = refine(s0, q0)
    C = float(np.exp(logC))
    return s, q, C, rmse, (rmin, rmax)'''
    nb.cells.append(new_code_cell(advanced_functions))


def add_heaps_execution_section(nb):
    """Add Heaps' law execution."""
    
    heaps_exec = '''# Execute Heaps' law analysis
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"file not found: {FILE_PATH}")

n_arr, V_arr = vocab_growth(FILE_PATH, stride=heaps_stride)
beta, K, r2, (n0, n1) = fit_heaps_logls(n_arr, V_arr, min_n=2000, tail_frac=0.7)

print("\\n=== heaps' law ===")
print(f"β (slope): {beta:.4f}")
print(f"K: {K:.4f}")
print(f"R^2 (log–log fit): {r2:.4f}")
print(f"fit range n in [{n0}, {n1}]   total tokens seen ~ {n_arr[-1]}   vocab ~ {V_arr[-1]}")'''
    nb.cells.append(new_code_cell(heaps_exec))


def add_zipf_mandelbrot_execution_section(nb):
    """Add Zipf-Mandelbrot execution."""
    
    zipf_exec = '''# Execute Zipf-Mandelbrot fitting
counts, freq = counts_from_file(FILE_PATH)
ranks = np.arange(1, len(counts)+1, dtype=np.int64)

rmin = max(1, min_rank_for_fit)
rmax = len(counts) if max_rank_for_fit is None else min(max_rank_for_fit, len(counts))
s, q, C, rmse_log, (rf0, rf1) = fit_zipf_mandelbrot(counts, rmin=rmin, rmax=rmax)

print("\\n=== zipf–mandelbrot ===")
print(f"s: {s:.4f}")
print(f"q: {q:.4f}")
print(f"C: {C:.4e}")
print(f"RMSE in log-frequency (fit ranks [{rf0}, {rf1}]): {rmse_log:.4f}")'''
    nb.cells.append(new_code_cell(zipf_exec))


def add_visualization_section(nb):
    """Add visualization plots."""
    
    viz_code = '''# Visualization plots

# Heaps' law plot
plt.figure(figsize=(7,5))
plt.loglog(n_arr, V_arr, marker='o', linestyle='none', markersize=2, label="Observed V(n)")
V_fit = K * (n_arr.astype(float) ** beta)
plt.loglog(n_arr, V_fit, linewidth=1.5, label=f"Fit: β={beta:.3f}, K={K:.2f}")
plt.xlabel("Tokens n (log)")
plt.ylabel("Vocabulary V(n) (log)")
plt.title("Heaps' Law on Quechua corpus")
plt.legend()
plt.tight_layout()
plt.show()

# Zipf curve with Zipf–Mandelbrot overlay
plt.figure(figsize=(7,5))
plt.loglog(ranks, counts, marker='o', linestyle='none', markersize=2, label="Empirical counts")
r = ranks.astype(float)
model = C / ((r + q) ** s)
plt.loglog(r, model, linewidth=1.5, label=f"Zipf–Mandelbrot fit (s={s:.3f}, q={q:.1f})")
plt.xlabel("Rank (log)")
plt.ylabel("Frequency (log)")
plt.title("Zipf curve with Zipf–Mandelbrot fit")
plt.legend()
plt.tight_layout()
plt.show()'''
    nb.cells.append(new_code_cell(viz_code))


def add_rolling_heaps_section(nb):
    """Add rolling Heaps exponent analysis."""
    
    rolling_code = '''# Rolling Heaps exponent analysis
win = 200

x = np.log(n_arr.astype(float))
y = np.log(V_arr.astype(float))

roll_beta = np.full_like(x, np.nan, dtype=float)
for i in range(win, len(x)):
    xs, ys = x[i-win:i], y[i-win:i]
    slope, _ = np.polyfit(xs, ys, 1)
    roll_beta[i] = slope

print("tail rolling β (last 5):", np.round(roll_beta[-5:], 3))'''
    nb.cells.append(new_code_cell(rolling_code))


def add_save_results_section(nb):
    """Add optional save results section."""
    
    save_code = '''# Optional: save results
# Uncomment to save analysis results

# Save Heaps' Law parameters
# heaps_results = {
#     'beta': float(beta),
#     'K': float(K),
#     'r2': float(r2),
#     'total_tokens': int(n_arr[-1]),
#     'vocab_size': int(V_arr[-1])
# }
# import json
# with open(os.path.join(DATA_FOLDER, 'heaps_law_results.json'), 'w') as f:
#     json.dump(heaps_results, f, indent=2)

# Save Zipf-Mandelbrot parameters
# zipf_mandelbrot_results = {
#     's': float(s),
#     'q': float(q),
#     'C': float(C),
#     'rmse_log': float(rmse_log),
#     'rank_range': [int(rf0), int(rf1)]
# }
# with open(os.path.join(DATA_FOLDER, 'zipf_mandelbrot_results.json'), 'w') as f:
#     json.dump(zipf_mandelbrot_results, f, indent=2)

# Save vocabulary growth data
# vocab_growth_df = pd.DataFrame({
#     'token_count': n_arr,
#     'vocab_size': V_arr
# })
# vocab_growth_df.to_csv(os.path.join(DATA_FOLDER, 'vocab_growth_heaps.csv'), index=False)

# Save word frequency distribution
# freq_df = pd.DataFrame({
#     'word': list(freq.keys()),
#     'frequency': list(freq.values())
# })
# freq_df = freq_df.sort_values('frequency', ascending=False)
# freq_df.to_csv(os.path.join(DATA_FOLDER, 'word_frequencies.csv'), index=False)'''
    nb.cells.append(new_code_cell(save_code))


def main():
    """Build and save the refactored notebook."""
    nb = create_notebook()
    
    # Add sections in order
    add_header_section(nb)
    add_imports_section(nb)
    add_config_section(nb)
    add_zipf_morphemes_section(nb)
    add_corpus_analysis_section(nb)
    add_advanced_modeling_section(nb)
    add_heaps_execution_section(nb)
    add_zipf_mandelbrot_execution_section(nb)
    add_visualization_section(nb)
    add_rolling_heaps_section(nb)
    add_save_results_section(nb)
    
    # Save notebook
    output_path = "analysis_refactored.ipynb"
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print(f"Refactored notebook saved to: {output_path}")


if __name__ == "__main__":
    main()

