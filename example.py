#!/usr/bin/env python
"""
Example: Quechua Morphological Segmentation

Demonstrates basic usage of the segmentation toolkit.
"""

import torch
from src import (
    to_graphemes,
    normalize_text,
    apply_boundaries,
    BiLSTMBoundary,
    BiLSTMWithPrior,
    HMMSuffixPrior,
    SuffixRejectionFilter,
    build_vocab
)


def main():
    print("=" * 60)
    print("Quechua Morphological Segmentation Demo")
    print("=" * 60)
    
    # Example words
    words = [
        "rikuchkani",   # riku-chka-ni (I am seeing)
        "wasiypi",      # wasi-y-pi (in my house)
        "llamkashanku", # llamka-sha-nku (they are working)
        "pikunas",      # pi-kuna-s (who-PL-TOPIC)
    ]
    
    gold_segmentations = [
        ["riku", "chka", "ni"],
        ["wasi", "y", "pi"],
        ["llamka", "sha", "nku"],
        ["pi", "kuna", "s"],
    ]
    
    # Step 1: Grapheme tokenization
    print("\n1. Grapheme Tokenization")
    print("-" * 40)
    for word in words:
        tokens = to_graphemes(word)
        print(f"  {word} → {tokens}")
    
    # Step 2: Build vocabulary
    print("\n2. Building Vocabulary")
    print("-" * 40)
    all_tokens = [to_graphemes(w) for w in words]
    stoi, itos = build_vocab(all_tokens)
    print(f"  Vocabulary size: {len(itos)} graphemes")
    print(f"  Sample: {itos[:10]}...")
    
    # Step 3: Train HMM prior (simplified demo)
    print("\n3. HMM Suffix Prior")
    print("-" * 40)
    hmm = HMMSuffixPrior(max_suffix_len=8)
    hmm.fit(gold_segmentations)
    
    for word in words[:2]:
        tokens = to_graphemes(word)
        probs = hmm.predict_probs(tokens)
        print(f"  {word}")
        print(f"    Tokens: {tokens}")
        print(f"    Prior P(boundary): {[f'{p:.2f}' for p in probs]}")
    
    # Step 4: Rejection filter
    print("\n4. Suffix Rejection Filter")
    print("-" * 40)
    
    # Extract suffix vocabulary from gold
    suffix_set = set()
    for morphs in gold_segmentations:
        for m in morphs[1:]:  # Skip root
            suffix_set.add(m.lower())
    
    filter = SuffixRejectionFilter(suffix_set)
    
    test_cases = [
        ["riku", "chka", "ni"],  # Valid
        ["riku", "xyz", "ni"],   # Invalid (xyz not in suffix vocab)
    ]
    
    for segments in test_cases:
        valid = filter.validate(segments)
        status = "✓ Valid" if valid else "✗ Rejected"
        print(f"  {segments} → {status}")
    
    # Step 5: Model architecture overview
    print("\n5. Model Architecture")
    print("-" * 40)
    
    model = BiLSTMBoundary(
        vocab_size=len(itos),
        emb_dim=64,
        hidden_size=128,
        num_layers=2,
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  BiLSTM Boundary Tagger")
    print(f"  Parameters: {total_params:,}")
    
    # Simulated forward pass
    tokens = to_graphemes("rikuchkani")
    x = torch.tensor([[stoi.get(t, 1) for t in tokens]])
    lengths = torch.tensor([len(tokens)])
    
    model.eval()
    with torch.no_grad():
        logits = model(x, lengths)
        probs = torch.sigmoid(logits).squeeze().tolist()
    
    print(f"\n  Example: rikuchkani")
    print(f"  Tokens: {tokens}")
    print(f"  P(boundary): {[f'{p:.2f}' for p in probs]}")
    
    # Step 6: Putting it all together
    print("\n6. Full Pipeline (Simulated)")
    print("-" * 40)
    
    # Simulate model prediction with correct boundaries
    simulated_labels = [0, 0, 0, 1, 0, 0, 1, 0]  # Boundaries after position 3 and 6
    
    segments = apply_boundaries(tokens, simulated_labels)
    print(f"  Input: rikuchkani")
    print(f"  Predicted segments: {segments}")
    print(f"  Gold: ['riku', 'chka', 'ni']")
    print(f"  Match: {'✓' if segments == ['riku', 'chka', 'ni'] else '✗'}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
