# Models Directory

This directory stores trained model checkpoints.

## Structure

Each model is saved in a subdirectory named by its configuration hash:

```
models/
├── {model_id}/
│   ├── model.pt      # PyTorch model state dict
│   └── vocab.json    # Vocabulary and metadata
└── ...
```

## Loading a Checkpoint

```python
from src import load_checkpoint, BiLSTMBoundary

checkpoint = load_checkpoint(model_id="abc123...", save_dir="models/")
if checkpoint:
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    
    model = BiLSTMBoundary(vocab_size=len(itos), ...)
    model.load_state_dict(checkpoint["model_state"])
```

## Pre-trained Models

Pre-trained model checkpoints are available on request. Due to file size 
constraints, they are not included in the repository by default.

| Model | Config Hash | Description |
|-------|-------------|-------------|
| BiLSTM (Grapheme) | `df7336ba5b7b6893` | Baseline grapheme BiLSTM |
| BiLSTM + HMM | `0dbdce1bde73fec1` | HMM prior with GPT-4o augmentation |
