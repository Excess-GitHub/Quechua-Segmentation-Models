"""
Evaluation utilities for Quechua morphological segmentation.

Metrics:
- Boundary F1 (precision, recall, F1 at boundary positions)
- Exact Match (full segmentation accuracy)
- Split-count metrics
"""

from typing import List, Set, Tuple, Dict, Any
import numpy as np


# ==============================================================================
# Boundary Metrics
# ==============================================================================

def boundary_positions_from_segments(segments: List[str]) -> Set[int]:
    """
    Extract boundary positions from segment list.
    
    Boundaries are at character positions between segments.
    
    Args:
        segments: List of morpheme strings
        
    Returns:
        Set of boundary positions (0-indexed, after each character)
    """
    positions = set()
    pos = 0
    
    for i, seg in enumerate(segments[:-1]):
        pos += len(seg)
        positions.add(pos - 1)  # Position after last char of segment
    
    return positions


def boundary_positions_from_token_segments(
    token_segments: List[List[str]]
) -> Set[int]:
    """
    Extract boundary positions from token-level segments.
    
    Args:
        token_segments: List of token lists, one per morpheme
        
    Returns:
        Set of boundary positions (0-indexed, after each token)
    """
    positions = set()
    pos = 0
    
    for i, toks in enumerate(token_segments[:-1]):
        pos += len(toks)
        positions.add(pos - 1)
    
    return positions


def compute_boundary_prf(
    pred_positions: Set[int],
    gold_positions: Set[int]
) -> Tuple[float, float, float, int, int, int]:
    """
    Compute precision, recall, F1 from boundary position sets.
    
    Args:
        pred_positions: Predicted boundary positions
        gold_positions: Gold boundary positions
        
    Returns:
        Tuple of (precision, recall, f1, tp, fp, fn)
    """
    tp = len(pred_positions & gold_positions)
    fp = len(pred_positions - gold_positions)
    fn = len(gold_positions - pred_positions)
    
    # Handle edge cases
    if tp + fp == 0:
        precision = 1.0 if tp + fn == 0 else 0.0
    else:
        precision = tp / (tp + fp)
    
    if tp + fn == 0:
        recall = 1.0 if tp + fp == 0 else 0.0
    else:
        recall = tp / (tp + fn)
    
    if precision + recall == 0:
        f1 = 1.0 if (tp + fp + fn) == 0 else 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1, tp, fp, fn


def aggregate_boundary_metrics(
    all_tp: int,
    all_fp: int,
    all_fn: int
) -> Dict[str, float]:
    """
    Compute micro-averaged boundary metrics from aggregated counts.
    
    Args:
        all_tp: Total true positives
        all_fp: Total false positives
        all_fn: Total false negatives
        
    Returns:
        Dict with precision, recall, f1
    """
    if all_tp + all_fp == 0:
        precision = 1.0 if all_tp + all_fn == 0 else 0.0
    else:
        precision = all_tp / (all_tp + all_fp)
    
    if all_tp + all_fn == 0:
        recall = 1.0 if all_tp + all_fp == 0 else 0.0
    else:
        recall = all_tp / (all_tp + all_fn)
    
    if precision + recall == 0:
        f1 = 1.0 if (all_tp + all_fp + all_fn) == 0 else 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": all_tp,
        "fp": all_fp,
        "fn": all_fn
    }


# ==============================================================================
# Exact Match Metrics
# ==============================================================================

def segments_match(
    predicted: List[str],
    gold: List[str]
) -> bool:
    """Check if predicted segments exactly match gold."""
    return predicted == gold


def segments_match_any_variant(
    predicted: List[str],
    gold_variants: List[List[str]]
) -> bool:
    """Check if predicted segments match any gold variant."""
    return any(predicted == variant for variant in gold_variants)


def best_variant_match(
    predicted: List[str],
    gold_variants: List[List[str]],
    tokenize_fn=None
) -> Tuple[List[str], float, Dict]:
    """
    Find the gold variant with best F1 match to prediction.
    
    Args:
        predicted: Predicted segment list
        gold_variants: List of gold segment lists
        tokenize_fn: Optional function to tokenize segments for comparison
        
    Returns:
        Tuple of (best_variant, best_f1, metrics_dict)
    """
    pred_positions = boundary_positions_from_segments(predicted)
    
    best = None
    best_f1 = -1
    best_metrics = None
    
    for variant in gold_variants:
        gold_positions = boundary_positions_from_segments(variant)
        p, r, f1, tp, fp, fn = compute_boundary_prf(pred_positions, gold_positions)
        
        if f1 > best_f1:
            best_f1 = f1
            best = variant
            best_metrics = {
                "precision": p,
                "recall": r,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn
            }
    
    return best, best_f1, best_metrics


# ==============================================================================
# Split-Count Metrics
# ==============================================================================

def compute_split_count_metrics(
    predicted: List[str],
    gold_variants: List[List[str]]
) -> Dict[str, bool]:
    """
    Compute split-count accuracy metrics.
    
    Args:
        predicted: Predicted segment list
        gold_variants: List of gold segment lists
        
    Returns:
        Dict with exact, +1, -1, ±1 match flags
    """
    pred_count = len(predicted)
    gold_counts = [len(g) for g in gold_variants]
    
    return {
        "exact": any(pred_count == g for g in gold_counts),
        "plus1": any(pred_count == g + 1 for g in gold_counts),
        "minus1": any(pred_count == g - 1 for g in gold_counts),
        "pm1": any(abs(pred_count - g) <= 1 for g in gold_counts)
    }


# ==============================================================================
# Full Evaluation
# ==============================================================================

def evaluate_predictions(
    words: List[str],
    predictions: List[List[str]],
    gold_variants_list: List[List[List[str]]],
    tokenize_fn=None
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of segmentation predictions.
    
    Args:
        words: List of input words
        predictions: List of predicted segment lists
        gold_variants_list: List of gold variant lists for each word
        tokenize_fn: Optional tokenization function for boundary comparison
        
    Returns:
        Evaluation results dict
    """
    results = {
        "n_words": len(words),
        "exact_matches": 0,
        "micro_tp": 0,
        "micro_fp": 0,
        "micro_fn": 0,
        "word_f1s": [],
        "split_exact": 0,
        "split_plus1": 0,
        "split_minus1": 0,
        "split_pm1": 0,
        "per_word": []
    }
    
    for word, pred, golds in zip(words, predictions, gold_variants_list):
        # Exact match
        is_exact = segments_match_any_variant(pred, golds)
        results["exact_matches"] += int(is_exact)
        
        # Boundary metrics against best variant
        pred_positions = boundary_positions_from_segments(pred)
        
        best_f1 = -1
        best_metrics = None
        
        for gold in golds:
            gold_positions = boundary_positions_from_segments(gold)
            p, r, f1, tp, fp, fn = compute_boundary_prf(pred_positions, gold_positions)
            
            if f1 > best_f1:
                best_f1 = f1
                best_metrics = {"p": p, "r": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        
        if best_metrics:
            results["micro_tp"] += best_metrics["tp"]
            results["micro_fp"] += best_metrics["fp"]
            results["micro_fn"] += best_metrics["fn"]
            results["word_f1s"].append(best_metrics["f1"])
        
        # Split-count metrics
        split_metrics = compute_split_count_metrics(pred, golds)
        results["split_exact"] += int(split_metrics["exact"])
        results["split_plus1"] += int(split_metrics["plus1"])
        results["split_minus1"] += int(split_metrics["minus1"])
        results["split_pm1"] += int(split_metrics["pm1"])
        
        # Per-word record
        results["per_word"].append({
            "word": word,
            "prediction": pred,
            "gold_variants": golds,
            "exact_match": is_exact,
            "boundary_f1": best_f1,
            "split_metrics": split_metrics
        })
    
    # Compute aggregate metrics
    n = results["n_words"]
    
    results["exact_match_rate"] = results["exact_matches"] / n if n > 0 else 0
    results["boundary_metrics"] = aggregate_boundary_metrics(
        results["micro_tp"], results["micro_fp"], results["micro_fn"]
    )
    results["macro_f1"] = np.mean(results["word_f1s"]) if results["word_f1s"] else 0
    
    results["split_exact_rate"] = results["split_exact"] / n if n > 0 else 0
    results["split_plus1_rate"] = results["split_plus1"] / n if n > 0 else 0
    results["split_minus1_rate"] = results["split_minus1"] / n if n > 0 else 0
    results["split_pm1_rate"] = results["split_pm1"] / n if n > 0 else 0
    
    return results


def print_evaluation_summary(results: Dict[str, Any], name: str = "Model"):
    """Print formatted evaluation summary."""
    print(f"\n{'=' * 60}")
    print(f"Evaluation Results: {name}")
    print(f"{'=' * 60}")
    print(f"Words evaluated: {results['n_words']}")
    print(f"\nExact Match: {results['exact_match_rate']:.4f} ({results['exact_matches']}/{results['n_words']})")
    
    bm = results['boundary_metrics']
    print(f"\nBoundary Metrics (micro):")
    print(f"  Precision: {bm['precision']:.4f}")
    print(f"  Recall:    {bm['recall']:.4f}")
    print(f"  F1:        {bm['f1']:.4f}")
    
    print(f"\nSplit-Count Metrics:")
    print(f"  Exact:  {results['split_exact_rate']:.4f}")
    print(f"  +1:     {results['split_plus1_rate']:.4f}")
    print(f"  -1:     {results['split_minus1_rate']:.4f}")
    print(f"  ±1:     {results['split_pm1_rate']:.4f}")
    print(f"{'=' * 60}\n")


# ==============================================================================
# Cross-Validation Utilities
# ==============================================================================

def compute_cv_summary(fold_results: List[Dict]) -> Dict[str, Any]:
    """
    Compute summary statistics across CV folds.
    
    Args:
        fold_results: List of per-fold result dicts
        
    Returns:
        Summary dict with means and stds
    """
    metrics = {}
    
    for key in ["exact_match_rate", "macro_f1"]:
        values = [r[key] for r in fold_results if key in r]
        if values:
            metrics[f"{key}_mean"] = np.mean(values)
            metrics[f"{key}_std"] = np.std(values)
    
    # Boundary F1
    f1s = [r["boundary_metrics"]["f1"] for r in fold_results if "boundary_metrics" in r]
    if f1s:
        metrics["boundary_f1_mean"] = np.mean(f1s)
        metrics["boundary_f1_std"] = np.std(f1s)
    
    return metrics


def print_cv_summary(fold_results: List[Dict], name: str = "Model"):
    """Print CV summary across folds."""
    print(f"\n{'=' * 60}")
    print(f"Cross-Validation Summary: {name}")
    print(f"{'=' * 60}")
    
    for i, r in enumerate(fold_results, 1):
        em = r.get("exact_match_rate", 0)
        f1 = r.get("boundary_metrics", {}).get("f1", 0)
        print(f"  Fold {i}: EM={em:.4f}, B-F1={f1:.4f}")
    
    summary = compute_cv_summary(fold_results)
    
    print(f"\nMean ± Std over {len(fold_results)} folds:")
    if "exact_match_rate_mean" in summary:
        print(f"  Exact Match: {summary['exact_match_rate_mean']:.4f} ± {summary['exact_match_rate_std']:.4f}")
    if "boundary_f1_mean" in summary:
        print(f"  Boundary F1: {summary['boundary_f1_mean']:.4f} ± {summary['boundary_f1_std']:.4f}")
    
    print(f"{'=' * 60}\n")
