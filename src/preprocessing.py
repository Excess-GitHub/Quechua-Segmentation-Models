"""
Preprocessing utilities for Quechua morphological segmentation.

Handles text normalization, grapheme tokenization, and boundary label generation.
"""

import re
import string
import unicodedata
from typing import List, Tuple, Set

# ==============================================================================
# Text Normalization
# ==============================================================================

APOSTROPHE_CHARS = {"'", "'", "ʼ", "‛", "`"}
STD_APOS = "\u02BC"  # Modifier letter apostrophe
EXTRA_PUNCT = "±，""''"
DELETE_TABLE = str.maketrans("", "", string.punctuation + EXTRA_PUNCT)


def normalize_text(s: str) -> str:
    """
    Normalize text for Quechua processing.
    
    Steps:
    1. NFC Unicode normalization
    2. Lowercase
    3. Unify apostrophe variants to standard form
    4. Strip punctuation
    
    Args:
        s: Input string
        
    Returns:
        Normalized string
    """
    s = unicodedata.normalize("NFC", str(s)).lower()
    s = "".join(STD_APOS if ch in APOSTROPHE_CHARS else ch for ch in s)
    s = s.translate(DELETE_TABLE).strip()
    return s


# ==============================================================================
# Grapheme Tokenization
# ==============================================================================

# Quechua grapheme inventory (34 symbols)
# Includes digraphs and trigraphs for aspirated/ejective stops
QUECHUA_MULTIGRAPHS = [
    # Ejective consonants (with apostrophe)
    "ch" + STD_APOS, "k" + STD_APOS, "p" + STD_APOS, "q" + STD_APOS, "t" + STD_APOS,
    # Digraphs
    "ch", "ph", "qh", "kh", "ll", "rr", "sh",
]

MG_SET = set(QUECHUA_MULTIGRAPHS)
MAX_MG_LEN = max((len(mg) for mg in QUECHUA_MULTIGRAPHS), default=1)

# Simple grapheme list (for character-level fallback)
GRAPHEMES = [
    "ch", "ll", "rr", "tr", "kw", "ph", "qh", "kh", "sh",
    "a", "b", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "ñ", "o", "p", "q",
    "r", "s", "t", "u", "v", "w", "x", "y"
]

GRAPHEME_PATTERN = re.compile("|".join(sorted(GRAPHEMES, key=len, reverse=True)))


def to_graphemes(s: str, normalize: bool = True) -> List[str]:
    """
    Tokenize string into Quechua graphemes using greedy longest-match.
    
    Respects digraphs (ch, ll, ph, etc.) and trigraphs (ejectives like chʼ).
    Falls back to single Unicode grapheme clusters for unmatched characters.
    
    Args:
        s: Input string (word or morpheme)
        normalize: Whether to apply text normalization first
        
    Returns:
        List of grapheme tokens
    """
    if normalize:
        s = normalize_text(s)
    
    tokens = []
    i = 0
    n = len(s)
    
    while i < n:
        match = None
        # Try longest match first
        for length in range(MAX_MG_LEN, 1, -1):
            if i + length <= n:
                candidate = s[i:i+length]
                if candidate in MG_SET:
                    match = candidate
                    break
        
        if match:
            tokens.append(match)
            i += len(match)
        else:
            # Fallback to Unicode grapheme cluster
            import regex
            m = regex.match(r"\X", s[i:])
            if m:
                g = m.group(0)
                tokens.append(g)
                i += len(g)
            else:
                tokens.append(s[i])
                i += 1
    
    return tokens


def tokenize_morphemes(morphemes: List[str]) -> List[List[str]]:
    """
    Tokenize a list of morphemes into grapheme sequences.
    
    Args:
        morphemes: List of morpheme strings
        
    Returns:
        List of grapheme token lists, one per morpheme
    """
    return [to_graphemes(m) for m in morphemes]


# ==============================================================================
# Boundary Labels
# ==============================================================================

def get_boundary_labels(tokens: List[str], morph_tokens: List[List[str]]) -> List[int]:
    """
    Generate binary boundary labels for token-level boundary prediction.
    
    A position i has label 1 if there's a morpheme boundary after token i.
    The last position always has label 0 (no boundary at end of word).
    
    Args:
        tokens: Grapheme tokens for the full word
        morph_tokens: List of grapheme token lists, one per morpheme
        
    Returns:
        List of binary labels (0 or 1) of same length as tokens
    """
    labels = [0] * len(tokens)
    idx = 0
    
    for mt in morph_tokens[:-1]:  # Skip last morpheme (no boundary after it)
        idx += len(mt)
        if 0 < idx <= len(tokens):
            labels[idx - 1] = 1
            
    return labels


def boundary_positions_from_labels(labels: List[int], length: int = None) -> Set[int]:
    """
    Convert boundary labels to set of boundary positions.
    
    Args:
        labels: Binary boundary labels
        length: Optional word length (uses len(labels) if not provided)
        
    Returns:
        Set of positions where boundaries occur
    """
    if not labels:
        return set()
    
    if length is None:
        length = len(labels)
    
    upper = min(length - 1, len(labels))
    return {i for i in range(upper) if labels[i] == 1}


def apply_boundaries(tokens: List[str], labels: List[int]) -> List[str]:
    """
    Apply boundary labels to split tokens into morpheme segments.
    
    Args:
        tokens: Grapheme tokens
        labels: Binary boundary labels
        
    Returns:
        List of morpheme strings
    """
    segments = []
    current = []
    
    for i, tok in enumerate(tokens):
        current.append(tok)
        if i < len(labels) and labels[i] == 1:
            segments.append("".join(current))
            current = []
    
    if current:
        segments.append("".join(current))
    
    return segments


# ==============================================================================
# CV Pattern Utilities
# ==============================================================================

VOWELS = set("aeiou")


def grapheme_to_cv(grapheme: str) -> str:
    """Convert a grapheme to C (consonant) or V (vowel)."""
    return "V" if grapheme.lower() in VOWELS else "C"


def tokens_to_cv_pattern(tokens: List[str]) -> str:
    """Convert token sequence to CV pattern string."""
    return "".join(grapheme_to_cv(t) for t in tokens)


def morphemes_to_cv_patterns(morph_tokens: List[List[str]]) -> List[str]:
    """Convert morpheme token lists to CV pattern strings."""
    return [tokens_to_cv_pattern(mt) for mt in morph_tokens]


# ==============================================================================
# Vocabulary Building
# ==============================================================================

PAD = "<PAD>"
UNK = "<UNK>"


def build_vocab(sequences: List[List[str]]) -> Tuple[dict, List[str]]:
    """
    Build vocabulary from token sequences.
    
    Args:
        sequences: List of token sequences
        
    Returns:
        Tuple of (token_to_idx dict, idx_to_token list)
    """
    tokens = {t for seq in sequences for t in seq}
    itos = [PAD, UNK] + sorted(tokens)
    stoi = {t: i for i, t in enumerate(itos)}
    return stoi, itos


def encode_sequence(seq: List[str], stoi: dict) -> List[int]:
    """
    Encode token sequence to integer IDs.
    
    Args:
        seq: Token sequence
        stoi: Token-to-index mapping
        
    Returns:
        List of integer IDs
    """
    unk_idx = stoi.get(UNK, 1)
    return [stoi.get(t, unk_idx) for t in seq]
