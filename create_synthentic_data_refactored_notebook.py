#!/usr/bin/env python3
"""
Script to generate a refactored version of the synthentic_data notebook.
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
        "# Synthetic Data Generation for Quechua Morphology Parser\n\n"
        "Generates synthetic morphological segmentation data using GPT models. "
        "Part 1: Data analysis and gold standard creation. "
        "Part 2: Synthetic data generation with few-shot learning."
    ))


def add_imports_section(nb):
    """Add all imports in a single organized cell."""
    imports_code = '''# Core libraries
import os
import ast
import time
import random
from collections import Counter
from pathlib import Path

# Data handling
import pandas as pd
import regex as re

# Unicode normalization
import unicodedata
from ftfy import fix_text

# API
from dotenv import load_dotenv
from openai import OpenAI
from openai import RateLimitError, APIError, APITimeoutError, APIConnectionError
from tqdm import tqdm'''
    nb.cells.append(new_code_cell(imports_code))


def add_config_section(nb):
    """Add configuration constants."""
    config_code = '''# Paths
DATA_FOLDER = "data"

# Input files
CORPUS_FILE = os.path.join(DATA_FOLDER, "qu_merged_dump.txt")
GOLD_DF_FILE = os.path.join(DATA_FOLDER, "Sue_kalt.parquet")
CLEANED_DF_FILE = os.path.join(DATA_FOLDER, "cleaned_data_df.csv")

# Output files
GOLD_OUTPUT_FILE = os.path.join(DATA_FOLDER, "gold_df_common_words.csv")
CLEANED_OUTPUT_FILE = os.path.join(DATA_FOLDER, "cleaned_data_df_common_words.csv")
COMMON_WORDS_OUTPUT_FILE = os.path.join(DATA_FOLDER, "word_analysis_gold.csv")
OUTPUT_FILE_GPT4O = os.path.join(DATA_FOLDER, "gpt4o_synthetic_segmentations.csv")
OUTPUT_FILE_GPT5MINI = os.path.join(DATA_FOLDER, "gpt5mini_synthetic_segmentations.csv")
GOLD_DATA_FILE = os.path.join(DATA_FOLDER, "word_analysis_gold.csv")

# Analysis parameters
RARE_WORD_RANK_THRESHOLD = 100000
LOWERCASE = True
KEEP_APOSTROPHES = True

# API parameters
load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODELS_TO_PROCESS = ["gpt-4o", "gpt-5-mini"]
NUM_FEW_SHOT_EXAMPLES = 37
WORDS_TO_PROCESS_LIMIT = 5'''
    nb.cells.append(new_code_cell(config_code))


def add_graphemes_section(nb):
    """Add grapheme definitions and tokenization functions."""
    
    graphemes_code = '''# Quechua graphemes
graphemes = [
    "ch","ll","rr","tr","kw","ph",
    "a","b","d","e","f","g","h","i","k","l","m","n","ñ","o","p","q",
    "r","s","t","u","v","w","x","y"
]

GRAPHEMES_BY_LEN = sorted(graphemes, key=len, reverse=True)
SINGLE_CHARS = {g for g in graphemes if len(g) == 1}

# Unicode normalization helper
CTRL_RE = re.compile(r'[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F\\x7F-\\x9F]')
def norm_unicode(x, form="NFC"):
    """Normalize unicode text."""
    if pd.isna(x):
        return x
    s = x.decode("utf-8", "replace") if isinstance(x, (bytes, bytearray)) else str(x)
    s = fix_text(s)
    s = CTRL_RE.sub('', s)
    s = unicodedata.normalize(form, s)
    s = s.replace('\\u00A0', ' ')
    s = re.sub(r'\\s+', ' ', s).strip()
    return s

def tokenize_graphemes(word: str):
    """Greedy longest-match tokenizer over allowed graphemes."""
    if not isinstance(word, str):
        return None
    w = word.strip()
    if LOWERCASE:
        w = w.lower()

    if "'" in w or "'" in w:
        return None

    i = 0
    toks = []
    n = len(w)
    while i < n:
        matched = False
        for g in GRAPHEMES_BY_LEN:
            L = len(g)
            if i + L <= n and w[i:i+L] == g:
                toks.append(g)
                i += L
                matched = True
                break
        if not matched:
            return None
    return toks

def is_valid_grapheme_word(word: str) -> bool:
    """Check if word can be fully segmented into allowed graphemes."""
    toks = tokenize_graphemes(word)
    return toks is not None

def first_four_graphemes_root(word: str) -> str:
    """Corpus root: concatenation of first 4 graphemes."""
    toks = tokenize_graphemes(word)
    if toks is None or len(toks) == 0:
        return ''
    root = ''.join(toks[:4])
    return root'''
    nb.cells.append(new_code_cell(graphemes_code))


def add_helper_functions_section(nb):
    """Add helper functions for data processing."""
    
    helpers_code = '''def safe_first_segment(row, prefer_list_col="Morph_split", fallback_str_col="Morph_split_str"):
    """Extract first segment (root) from row, handling various formats."""
    if prefer_list_col in row:
        val = row[prefer_list_col]
        if isinstance(val, list) and len(val) > 0:
            return str(val[0]).strip()
        if isinstance(val, str):
            s = val.strip()
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return str(parsed[0]).strip()
            except Exception:
                pass

    if fallback_str_col in row:
        s = row[fallback_str_col]
        if isinstance(s, str) and s.strip():
            return s.strip().split()[0]

    return ''

def robust_first_segment(row, prefer_list_col="Morph_split", fallback_str_col="Morph_split_str", alt_morph_col="morph"):
    """Extract root as first segment, with multiple fallbacks."""
    if prefer_list_col in row:
        val = row[prefer_list_col]
        if isinstance(val, list) and val:
            return str(val[0]).strip()
        if isinstance(val, str):
            s = val.strip()
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list) and parsed:
                    return str(parsed[0]).strip()
            except Exception:
                if s:
                    return s.split()[0].strip()

    if fallback_str_col in row:
        s = row[fallback_str_col]
        if isinstance(s, str) and s.strip():
            return s.strip().split()[0]

    if alt_morph_col in row:
        m = row[alt_morph_col]
        if isinstance(m, str) and m.strip():
            return m.replace('-', ' ').strip().split()[0]

    return ''

def process_corpus(file_path):
    """Read corpus, tokenize, and calculate word frequencies (valid grapheme words only)."""
    print(f"processing corpus file: {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"corpus file not found at: {file_path}")

    TOKEN_RE = re.compile(r"[^\\W\\d_]+(?:[''][^\\W\\d_]+)?", flags=re.UNICODE) if KEEP_APOSTROPHES \\
                else re.compile(r"[^\\W\\d_]+", flags=re.UNICODE)

    def iter_valid_tokens_from_file(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if LOWERCASE:
                    line = line.lower()
                for m in TOKEN_RE.finditer(line):
                    tok = m.group(0)
                    if is_valid_grapheme_word(tok):
                        yield tok

    freq = Counter(iter_valid_tokens_from_file(file_path))

    if not freq:
        print("hmm, got zero tokens after grapheme filtering")
        return {}, {}
        
    print(f"corpus processed. total unique valid grapheme-words: {len(freq):,}")

    sorted_words = [word for word, count in freq.most_common()]
    rank_map = {word: i + 1 for i, word in enumerate(sorted_words)}
    
    return dict(freq), rank_map'''
    nb.cells.append(new_code_cell(helpers_code))


def add_part1_analysis_section(nb):
    """Add Part 1: Data analysis and gold standard creation."""
    
    part1_code = '''# Part 1: Data Analysis and Gold Standard Creation

# Step 1: Process corpus
corpus_freq, corpus_rank = process_corpus(CORPUS_FILE)

# Step 2: Load dataframes
gold_df = pd.read_parquet(GOLD_DF_FILE)
gold_df['Word'] = gold_df['word']
gold_df['morph'] = gold_df['morph'].str.replace('-', ' ')
gold_df['Morph_split_str'] = gold_df['morph']
gold_df['Morph_split'] = gold_df['morph'].str.split(' ')
gold_df = gold_df[['Word', 'Morph_split', 'Morph_split_str']]

cleaned_df = pd.read_csv(CLEANED_DF_FILE, encoding='windows-1252')

# Extract word sets
gold_words = set(gold_df['Word'].dropna().unique())
cleaned_words = set(cleaned_df['Word'].dropna().unique())
corpus_words = set(corpus_freq.keys())

print("\\n" + "="*50)
print("ANALYSIS RESULTS")
print("="*50 + "\\n")

# Step 3: Corpus coverage analysis
print("--- 1. corpus coverage analysis (surface forms) ---")
gold_in_corpus = gold_words.intersection(corpus_words)
coverage_percentage = (len(gold_in_corpus) / len(gold_words)) * 100 if gold_words else 0
print(f"[{GOLD_DF_FILE}]: found {len(gold_in_corpus):,} / {len(gold_words):,} words in corpus ({coverage_percentage:.2f}% coverage)\\n")

cleaned_in_corpus = cleaned_words.intersection(corpus_words)
coverage_percentage = (len(cleaned_in_corpus) / len(cleaned_words)) * 100 if cleaned_words else 0
print(f"[{CLEANED_DF_FILE}]: found {len(cleaned_in_corpus):,} / {len(cleaned_words):,} words in corpus ({coverage_percentage:.2f}% coverage)\\n")

# Step 4: Dataset incongruity analysis
print("--- 2. dataset incongruity analysis (surface forms) ---")
words_in_common = gold_words.intersection(cleaned_words)
words_only_in_gold = gold_words.difference(cleaned_words)
words_only_in_cleaned = cleaned_words.difference(gold_words)
common_and_in_corpus = words_in_common.intersection(corpus_words)

print(f"words common to both datasets: {len(words_in_common):,}")
print(f"words in the corpus and common to both datasets: {len(common_and_in_corpus):,}")
print(f"words only in '{GOLD_DF_FILE}': {len(words_only_in_gold):,}")
print(f"words only in '{CLEANED_DF_FILE}': {len(words_only_in_cleaned):,}\\n")

# Step 4b: Root-level analysis
print("--- 2b. root-level analysis ---")

corpus_roots = set()
for w in corpus_words:
    r = first_four_graphemes_root(w)
    if r:
        corpus_roots.add(r)

gold_df = gold_df.copy()
gold_df['Root'] = gold_df.apply(lambda row: safe_first_segment(row, "Morph_split", "Morph_split_str"), axis=1)
gold_roots = set([r for r in gold_df['Root'].dropna().map(str).map(str.strip) if r])

cleaned_df = cleaned_df.copy()
if 'Morph_split_str' not in cleaned_df.columns:
    if 'Morph_split' in cleaned_df.columns:
        def to_str_split(val):
            if isinstance(val, list):
                return ' '.join(map(str, val))
            if isinstance(val, str):
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        return ' '.join(map(str, parsed))
                except Exception:
                    return val
            return ''
        cleaned_df['Morph_split_str'] = cleaned_df['Morph_split'].apply(to_str_split)
    else:
        cleaned_df['Morph_split_str'] = ''

cleaned_df['Root'] = cleaned_df.apply(lambda row: safe_first_segment(row, "Morph_split", "Morph_split_str"), axis=1)
cleaned_roots = set([r for r in cleaned_df['Root'].dropna().map(str).map(str.strip) if r])

print(f"unique roots in corpus (first 4 graphemes): {len(corpus_roots):,}")
print(f"unique roots in {GOLD_DF_FILE} (first segment): {len(gold_roots):,}")
print(f"unique roots in {CLEANED_DF_FILE} (first segment): {len(cleaned_roots):,}")

roots_gold_cleaned = gold_roots.intersection(cleaned_roots)
roots_gold_corpus = gold_roots.intersection(corpus_roots)
roots_cleaned_corpus = cleaned_roots.intersection(corpus_roots)
roots_all_three = gold_roots.intersection(cleaned_roots).intersection(corpus_roots)

print(f"overlapping roots (gold ∩ cleaned): {len(roots_gold_cleaned):,}")
print(f"overlapping roots (gold ∩ corpus): {len(roots_gold_corpus):,}")
print(f"overlapping roots (cleaned ∩ corpus): {len(roots_cleaned_corpus):,}")
print(f"overlapping roots (gold ∩ cleaned ∩ corpus): {len(roots_all_three):,}\\n")

# Step 5: Rarity analysis
print(f"--- 3. rarity analysis (threshold: top {RARE_WORD_RANK_THRESHOLD:,} words) ---")
rare_words_in_gold = {word for word in gold_words if corpus_rank.get(word, float('inf')) > RARE_WORD_RANK_THRESHOLD}
print(f"[{GOLD_DF_FILE}]: {len(rare_words_in_gold):,} words are 'rare' (rank > {RARE_WORD_RANK_THRESHOLD:,})")

rare_words_in_cleaned = {word for word in cleaned_words if corpus_rank.get(word, float('inf')) > RARE_WORD_RANK_THRESHOLD}
print(f"[{CLEANED_DF_FILE}]: {len(rare_words_in_cleaned):,} words are 'rare' (rank > {RARE_WORD_RANK_THRESHOLD:,})\\n")

# Step 6: Coverage of non-rare words
print("--- 4. coverage of non-rare words ---")
common_gold = gold_words - rare_words_in_gold
common_cleaned = cleaned_words - rare_words_in_cleaned

common_gold_in_corpus = common_gold.intersection(corpus_words)
coverage_perc = (len(common_gold_in_corpus) / len(common_gold)) * 100 if common_gold else 0
print(f"[{GOLD_DF_FILE}]: of its {len(common_gold):,} non-rare words, {len(common_gold_in_corpus):,} ({coverage_perc:.2f}%) are in the corpus")

common_cleaned_in_corpus = common_cleaned.intersection(corpus_words)
coverage_perc = (len(common_cleaned_in_corpus) / len(common_cleaned)) * 100 if common_cleaned else 0
print(f"[{CLEANED_DF_FILE}]: of its {len(common_cleaned):,} non-rare words, {len(common_cleaned_in_corpus):,} ({coverage_perc:.2f}%) are in the corpus\\n")

# Step 7: Remove rare words and save
print("--- 5. removing rare words and saving new CSVs ---")

if not gold_df.empty:
    filtered_gold_df = gold_df[~gold_df['Word'].isin(rare_words_in_gold)]
    filtered_gold_df.to_csv(GOLD_OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"removed {len(rare_words_in_gold)} rare words from '{GOLD_DF_FILE}'")
    print(f"-> saved {len(filtered_gold_df)} rows to '{GOLD_OUTPUT_FILE}'\\n")

if not cleaned_df.empty:
    filtered_cleaned_df = cleaned_df[~cleaned_df['Word'].isin(rare_words_in_cleaned)]
    filtered_cleaned_df.to_csv(CLEANED_OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"removed {len(rare_words_in_cleaned)} rare words from '{CLEANED_DF_FILE}'")
    print(f"-> saved {len(filtered_cleaned_df)} rows to '{CLEANED_OUTPUT_FILE}'\\n")

# Step 2c: Word-level gold (common words with common roots)
print("--- 2c. word-level gold (common words with common roots) ---")

def _seg_str_from_row(row):
    if 'Morph_split' in row:
        ms = row['Morph_split']
        if isinstance(ms, list):
            s = ' '.join(map(str, ms)).strip()
            if s: return s
        if isinstance(ms, str):
            s = ms.strip()
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list) and parsed:
                    s2 = ' '.join(map(str, parsed)).strip()
                    if s2: return s2
            except Exception:
                if s: return s
    if 'Morph_split_str' in row and isinstance(row['Morph_split_str'], str):
        s = row['Morph_split_str'].strip()
        if s: return s
    if 'morph' in row and isinstance(row['morph'], str):
        s = row['morph'].replace('-', ' ').strip()
        if s: return s
    return ''

def _first_nonempty_map(df, value_col):
    tmp = (
        df[['Word', value_col]]
        .copy()
        .dropna(subset=['Word'])
    )
    tmp['Word'] = tmp['Word'].astype(str).str.strip()
    tmp[value_col] = tmp[value_col].astype(str).str.strip()
    tmp = tmp[tmp['Word'] != '']
    tmp = tmp[tmp[value_col] != '']
    return tmp.drop_duplicates(subset=['Word']).set_index('Word')[value_col].to_dict()

gold_root_map = _first_nonempty_map(gold_df.rename(columns={'Root':'__Root'}), '__Root')
cleaned_root_map = _first_nonempty_map(cleaned_df.rename(columns={'Root':'__Root'}), '__Root')

if not gold_df.empty:
    _gdf = gold_df.copy()
    _gdf['__Seg'] = _gdf.apply(_seg_str_from_row, axis=1)
    gold_seg_map = _first_nonempty_map(_gdf, '__Seg')
else:
    gold_seg_map = {}

if not cleaned_df.empty:
    _cldf = cleaned_df.copy()
    _cldf['__Seg'] = _cldf.apply(_seg_str_from_row, axis=1)
    cleaned_seg_map = _first_nonempty_map(_cldf, '__Seg')
else:
    cleaned_seg_map = {}

words_all_three = gold_words.intersection(cleaned_words).intersection(corpus_words)
print(f"surface-overlap across all three datasets: {len(words_all_three):,} words")

rows = []
kept = 0
for w in words_all_three:
    c_root = first_four_graphemes_root(w) or ''
    r_gold = gold_root_map.get(w, '')
    r_clean = cleaned_root_map.get(w, '')

    if c_root and r_gold and r_clean and (c_root == r_gold == r_clean) and (c_root in roots_all_three):
        seg = cleaned_seg_map.get(w, '') or gold_seg_map.get(w, '')
        if seg:
            rows.append({'Word': w, 'Morph_split': seg})
            kept += 1

word_level_gold_df = pd.DataFrame(rows).sort_values('Word')
word_level_gold_df.to_csv(COMMON_WORDS_OUTPUT_FILE, index=False, encoding='utf-8')
print(f"-> saved {kept:,} rows to '{COMMON_WORDS_OUTPUT_FILE}' (columns: Word, Morph_split)\\n")'''
    nb.cells.append(new_code_cell(part1_code))


def add_part2_api_functions_section(nb):
    """Add Part 2: API functions for synthetic data generation."""
    
    api_functions = '''# Part 2: Synthetic Data Generation Using GPT Models

def load_all_data():
    """Load data files and identify words needing segmentation."""
    print("--- step 1: loading all data files ---")

    if not os.path.exists(GOLD_DATA_FILE):
        raise FileNotFoundError(f"gold data file not found: '{GOLD_DATA_FILE}'. need to run the previous script first")
    gold_df = pd.read_csv(GOLD_DATA_FILE)

    if 'Morph_split_str' not in gold_df.columns:
        gold_df['Morph_split_str'] = ''
    def _mk_str(val):
        if isinstance(val, list):
            return ' '.join(map(str, val))
        if isinstance(val, str):
            s = val.strip()
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return ' '.join(map(str, parsed))
            except Exception:
                return s
        return ''
    if 'Morph_split' in gold_df.columns:
        gold_df['Morph_split_str'] = gold_df['Morph_split'].apply(_mk_str)
    print(f"loaded {len(gold_df):,} 'gold' examples for few-shot learning")

    combined_df = pd.read_parquet(GOLD_DF_FILE)
    combined_df['Word'] = combined_df['word']
    combined_df['morph'] = combined_df['morph'].str.replace('-', ' ')
    combined_df['Morph_split_str'] = combined_df['morph']
    combined_df['Morph_split'] = combined_df['morph'].str.split(' ')
    combined_df = combined_df[['Word', 'Morph_split', 'Morph_split_str']]
    cleaned_df = pd.read_csv(CLEANED_DF_FILE, encoding='windows-1252')

    if 'Morph_split_str' not in combined_df.columns and 'Morph_split' in combined_df.columns:
        def _to_str_split(val):
            if isinstance(val, list):
                return ' '.join(map(str, val))
            if isinstance(val, str):
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        return ' '.join(map(str, parsed))
                except Exception:
                    return val
            return ''
        combined_df['Morph_split_str'] = combined_df['Morph_split'].apply(_to_str_split) if 'Morph_split' in combined_df.columns else ''

    if 'Morph_split_str' not in cleaned_df.columns and 'Morph_split' in cleaned_df.columns:
        def _to_str_split2(val):
            if isinstance(val, list):
                return ' '.join(map(str, val))
            if isinstance(val, str):
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        return ' '.join(map(str, parsed))
                except Exception:
                    return val
            return ''
        cleaned_df['Morph_split_str'] = cleaned_df['Morph_split'].apply(_to_str_split2) if 'Morph_split' in cleaned_df.columns else ''

    existing_words = set(combined_df['Word'].dropna()) | set(cleaned_df['Word'].dropna())
    print(f"found {len(existing_words):,} unique words across existing datasets")

    print("reading full corpus to find target words...")
    if not os.path.exists(CORPUS_FILE):
        raise FileNotFoundError(f"corpus file not found: {CORPUS_FILE}")
    TOKEN_RE = re.compile(r"[^\\W\\d_]+(?:[''][^\\W\\d_]+)?", flags=re.UNICODE)
    with open(CORPUS_FILE, "r", encoding="utf-8", errors="ignore") as f:
        corpus_text = f.read().lower()
    corpus_words_all = set(TOKEN_RE.findall(corpus_text))
    print(f"found {len(corpus_words_all):,} unique words in the corpus")

    corpus_roots = set()
    for w in corpus_words_all:
        r = first_four_graphemes_root(w)
        if r:
            corpus_roots.add(r)

    combined_roots = set()
    if not combined_df.empty:
        combined_df = combined_df.copy()
        combined_df['__root__'] = combined_df.apply(
            lambda row: robust_first_segment(row, "Morph_split", "Morph_split_str", "morph"), axis=1
        )
        combined_roots = set([r for r in combined_df['__root__'].dropna().map(str).map(str.strip) if r])

    cleaned_roots = set()
    if not cleaned_df.empty:
        cleaned_df = cleaned_df.copy()
        cleaned_df['__root__'] = cleaned_df.apply(
            lambda row: robust_first_segment(row, "Morph_split", "Morph_split_str", "morph"), axis=1
        )
        cleaned_roots = set([r for r in cleaned_df['__root__'].dropna().map(str).map(str.strip) if r])

    gold_roots = combined_roots

    common_roots_all_three = corpus_roots.intersection(gold_roots).intersection(cleaned_roots)
    print(f"roots common to all three datasets: {len(common_roots_all_three):,}")

    candidate_words = sorted(list(corpus_words_all - existing_words))
    print(f"-> initially identified {len(candidate_words):,} new corpus words (not in existing datasets)")

    words_to_segment = []
    for w in candidate_words:
        root = first_four_graphemes_root(w)
        if root and root in common_roots_all_three:
            words_to_segment.append(w)

    print(f"-> filtered to {len(words_to_segment):,} words whose roots are common to all three datasets\\n")

    return gold_df, words_to_segment

def construct_few_shot_prompt(target_word, gold_df, num_examples):
    """Create prompt with few-shot examples."""
    examples = gold_df.sample(n=min(num_examples, len(gold_df)), random_state=random.randint(0, 10_000))

    system_message = (
        "You are an expert in Quechua linguistics. Your task is to segment a given Quechua word into its constituent morphemes. "
        "The morphemes should be separated by spaces. Provide only the segmented output, with no additional explanation or commentary."
    )

    messages = [{"role": "system", "content": system_message}]
    for _, row in examples.iterrows():
        s = row.get('Morph_split_str', '')
        if not isinstance(s, str) or not s.strip():
            s = ''
            if 'Morph_split' in row and isinstance(row['Morph_split'], str):
                try:
                    parsed = ast.literal_eval(row['Morph_split'])
                    if isinstance(parsed, list):
                        s = ' '.join(map(str, parsed))
                except Exception:
                    s = row['Morph_split']
        messages.append({"role": "user", "content": str(row['Word'])})
        messages.append({"role": "assistant", "content": s})

    messages.append({"role": "user", "content": target_word})
    return messages

def get_model_params(model_name):
    """Get model-specific API parameters."""
    if "gpt-5" in model_name.lower() or "gpt5" in model_name.lower():
        return {
            "reasoning_effort": "minimal",
            "verbosity": "low",
        }
    elif "gpt-4o-mini" in model_name.lower() or "gpt-4o-mini" in model_name:
        return {
            "max_tokens": 50,
            "temperature": 0,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
    else:
        return {
            "max_tokens": 50,
            "temperature": 0,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }

def get_llm_segmentation(prompt_messages, model_name, retries=3, delay=5):
    """Call LLM API to get word segmentation, with rate limit handling."""
    def _retry_after_seconds(err, fallback):
        try:
            resp = getattr(err, "response", None)
            if resp and getattr(resp, "headers", None):
                ra = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
                if ra:
                    return float(ra)
        except Exception:
            pass
        return fallback

    api_params = get_model_params(model_name)
    api_params["model"] = model_name
    api_params["messages"] = prompt_messages

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(**api_params)
            return (response.choices[0].message.content or "").strip()

        except RateLimitError as e:
            base = delay * (2 ** attempt)
            wait = _retry_after_seconds(e, base) + random.uniform(0, 0.5)
            print(f"  [ratelimit] hit 429. waiting {wait:.2f}s before retry {attempt+1}/{retries}...")
            time.sleep(wait)

        except (APITimeoutError, APIConnectionError) as e:
            wait = delay * (2 ** attempt) + random.uniform(0, 0.5)
            print(f"  [transient] {type(e).__name__}: {e}. waiting {wait:.2f}s (retry {attempt+1}/{retries})...")
            time.sleep(wait)

        except APIError as e:
            status = getattr(e, "status_code", None)
            if status == 429:
                base = delay * (2 ** attempt)
                wait = _retry_after_seconds(e, base) + random.uniform(0, 0.5)
                print(f"  [api 429] waiting {wait:.2f}s before retry {attempt+1}/{retries}...")
                time.sleep(wait)
            elif status == 400:
                error_msg = str(e)
                if "parameter" in error_msg.lower() or "invalid" in error_msg.lower():
                    print(f"  [api parameter error] model '{model_name}' may not support some parameters")
                    print(f"  error: {error_msg}")
                    print(f"  maybe check get_model_params() and adjust for this model")
                    print(f"  might need to remove top_p, frequency_penalty, or presence_penalty")
                else:
                    print(f"  [api error 400]: {e}")
                break
            else:
                print(f"  [api error] {status}: {e}")
                break

        except Exception as e:
            print(f"  [unhandled error]: {e}")
            break

    return "[API_FAILED]"'''
    nb.cells.append(new_code_cell(api_functions))


def add_part2_main_section(nb):
    """Add Part 2: Main execution for synthetic data generation."""
    
    main_code = '''# Main execution: Generate synthetic data
if not os.environ.get("OPENAI_API_KEY"):
    print("error: OPENAI_API_KEY not set")
    print("need to set it before running")
else:
    print("="*70)
    print("SYNTHETIC DATA GENERATION FOR QUECHUA MORPHOLOGY")
    print("="*70)
    gold_df, words_to_segment = load_all_data()

    if WORDS_TO_PROCESS_LIMIT is not None:
        print(f"\\n--- applying processing limit: selecting {WORDS_TO_PROCESS_LIMIT} words randomly ---")
        if len(words_to_segment) > WORDS_TO_PROCESS_LIMIT:
            words_to_segment = random.sample(words_to_segment, WORDS_TO_PROCESS_LIMIT)
        else:
            print("limit is larger than the number of available words. processing all")

    model_output_map = {
        "gpt-4o": OUTPUT_FILE_GPT4O,
        "gpt-5-mini": OUTPUT_FILE_GPT5MINI
    }

    for model_name in MODELS_TO_PROCESS:
        if model_name not in model_output_map:
            print(f"unknown model '{model_name}', skipping...")
            continue
            
        output_file = model_output_map[model_name]
        print(f"\\n{'='*70}")
        print(f"processing {len(words_to_segment):,} words using '{model_name}'")
        print(f"{'='*70}")
        
        results = []
        for word in tqdm(words_to_segment, desc=f"segmenting with {model_name}"):
            prompt = construct_few_shot_prompt(word, gold_df, NUM_FEW_SHOT_EXAMPLES)
            segmented_word = get_llm_segmentation(prompt, model_name)
            results.append({
                'Original_Word': word,
                'Segmented_Morphemes': segmented_word,
                'Source': f'LLM_FewShot_{model_name}',
                'Model': model_name
            })

        print(f"\\n--- saving results for {model_name} ---")
        results_df = pd.DataFrame(results)
        results_df = results_df[results_df['Segmented_Morphemes'] != '[API_FAILED]']
        results_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"done! processed {len(results_df)} words with {model_name}")
        print(f"   saved to '{output_file}'")
        print(f"   failed calls: {len(results) - len(results_df)}")
    
    print(f"\\n{'='*70}")
    print("all done!")
    print(f"{'='*70}")
    print(f"got segmentations for {len(words_to_segment):,} words using {len(MODELS_TO_PROCESS)} models")
    print(f"files saved to {DATA_FOLDER}/")'''
    nb.cells.append(new_code_cell(main_code))


def main():
    """Build and save the refactored notebook."""
    nb = create_notebook()
    
    # Add sections in order
    add_header_section(nb)
    add_imports_section(nb)
    add_config_section(nb)
    add_graphemes_section(nb)
    add_helper_functions_section(nb)
    add_part1_analysis_section(nb)
    add_part2_api_functions_section(nb)
    add_part2_main_section(nb)
    
    # Save notebook
    output_path = "synthentic_data_refactored.ipynb"
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print(f"Refactored notebook saved to: {output_path}")


if __name__ == "__main__":
    main()

