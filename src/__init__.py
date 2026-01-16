"""
Quechua Morphological Segmentation

A toolkit for supervised morphological segmentation of Southern Quechua,
featuring neural architectures with linguistically-informed priors.
"""

from .preprocessing import (
    normalize_text,
    to_graphemes,
    tokenize_morphemes,
    get_boundary_labels,
    apply_boundaries,
    build_vocab,
    encode_sequence,
    PAD, UNK,
    VOWELS,
    GRAPHEMES,
    QUECHUA_MULTIGRAPHS
)

from .models import (
    BiLSTMBoundary,
    BiLSTMWithPrior,
    BiLSTMCRF,
    TransformerSegmenter,
    DecisionTreePrior,
    HMMSuffixPrior,
    SuffixRejectionFilter
)

from .evaluation import (
    compute_boundary_prf,
    evaluate_predictions,
    print_evaluation_summary,
    compute_cv_summary,
    print_cv_summary
)

from .training import (
    BoundaryDataset,
    Seq2SeqDataset,
    boundary_collate_fn,
    train_boundary_model,
    tune_threshold,
    save_checkpoint,
    load_checkpoint,
    generate_model_id,
    run_kfold_cv
)

__version__ = "0.1.0"
__author__ = "Anonymous"
