# Notebooks Directory

This directory contains the original Jupyter notebooks used for experimentation 
and development. The clean Python modules in `src/` were extracted from these 
notebooks.

## Notebook Descriptions

### Core Models

| Notebook | Description | Key Results |
|----------|-------------|-------------|
| `segmenter.ipynb` | BiLSTM with grapheme tokenization | 56.1% EM, 0.840 B-F1 |
| `segmenter-old.ipynb` | BiLSTM with character tokenization | 52.7% EM, 0.817 B-F1 |
| `segmenter-morfessor.ipynb` | BiLSTM + Morfessor features | 55.1% EM, 0.838 B-F1 |
| `stats.ipynb` | BiLSTM-CRF model | 84.9% CV EM |
| `transmorpher.ipynb` | Transformer Seq2Seq | 43.2% EM, 60.5% CV EM |

### Hybrid Models with Priors

| Notebook | Description | Key Results |
|----------|-------------|-------------|
| `DT-LSTM-MarkovFilter.ipynb` | BiLSTM + Decision Tree prior + rejection filter | 64.8% EM (with filter) |
| `Markov-LSTM-MarkovFilter.ipynb` | BiLSTM + HMM prior + rejection filter + LLM augmentation | **74.2% EM** (best) |

### Analysis

| Notebook | Description |
|----------|-------------|
| `analysis.ipynb` | Corpus statistics: Zipf's law, Heaps' law, Zipf-Mandelbrot fitting |

## Running the Notebooks

```bash
# Install Jupyter
pip install jupyter

# Start Jupyter
jupyter notebook notebooks/
```

## Note

These notebooks were used for exploration and may contain experimental code 
or debugging output. For production use, please use the clean `src/` modules.
