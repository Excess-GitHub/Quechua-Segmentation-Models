#!/usr/bin/env python3
"""
Script to generate a refactored version of the stats notebook.
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
        "# Stats: Outlier Removal Experiment\n\n"
        "Experimental analysis to test if removing statistical outliers from training data "
        "improves BiLSTM+CRF morphology parser performance. Compares two models: "
        "one trained on full data, one trained on filtered data (outliers removed)."
    ))


def add_imports_section(nb):
    """Add all imports in a single organized cell."""
    imports_code = '''# Core libraries
import os
import ast
import json
import hashlib
from pathlib import Path

# Data handling
import numpy as np
import pandas as pd

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

# Statistics
from scipy.stats import pearsonr, spearmanr

# ML
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF'''
    nb.cells.append(new_code_cell(imports_code))


def add_config_section(nb):
    """Add configuration constants."""
    config_code = '''# Paths
DATA_FOLDER = "data"
MODEL_NAME = "stats"
MODELS_FOLDER = f"models_{MODEL_NAME}"
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")'''
    nb.cells.append(new_code_cell(config_code))


def add_data_loading_section(nb):
    """Add data loading and preprocessing."""
    
    data_loading = '''# Load gold standard data
print("loading gold data...")
gold_df = pd.read_parquet(os.path.join(DATA_FOLDER, "Sue_kalt.parquet"))
gold_df['Word'] = gold_df['word']
gold_df['morph'] = gold_df['morph'].str.replace('-', ' ')
gold_df['Morph_split_str'] = gold_df['morph']
gold_df['Morph_split'] = gold_df['morph'].str.split(' ')
gold_df = gold_df[['Word', 'Morph_split', 'Morph_split_str']]
gold_df = gold_df.drop_duplicates(subset=['Word']).reset_index(drop=True)
gold_df = gold_df.dropna(subset=['Word'])
print(f"got {len(gold_df):,} examples")'''
    nb.cells.append(new_code_cell(data_loading))
    
    features = '''# Extract basic features
gold_df['num_morphemes'] = gold_df['Morph_split'].apply(len)
gold_df['word_len'] = gold_df['Word'].apply(len)'''
    nb.cells.append(new_code_cell(features))


def add_statistical_analysis_section(nb):
    """Add statistical analysis (correlation, regression, outlier detection)."""
    
    heatmap = '''# Heatmap of word length vs morpheme count
heatmap_data = gold_df.groupby(['word_len', 'num_morphemes']).size().unstack(fill_value=0)

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Heatmap of Word Length vs. Morpheme Count')
plt.xlabel('Number of Morphemes')
plt.ylabel('Word Length (Characters)')
plt.tight_layout()
plt.show()'''
    nb.cells.append(new_code_cell(heatmap))
    
    correlation = '''# Correlation analysis
x = gold_df['word_len']
y = gold_df['num_morphemes']

pearson_corr, pearson_p = pearsonr(x, y)
spearman_corr, spearman_p = spearmanr(x, y)

print(f"pearson correlation: {pearson_corr:.3f} (p={pearson_p:.3e})")
print(f"spearman correlation: {spearman_corr:.3f} (p={spearman_p:.3e})")
print("\\nstrong positive correlation means longer words tend to have more morphemes")
print("we'll use this relationship to find outliers")'''
    nb.cells.append(new_code_cell(correlation))
    
    linear_regression = '''# Linear regression to identify outliers
gold_df1 = gold_df.copy()
print(f"original size: {gold_df1.shape}")

X = gold_df1[['word_len']]
y = gold_df1['num_morphemes']

model = LinearRegression()
model.fit(X, y)
gold_df1['predicted'] = model.predict(X)
gold_df1['residual'] = gold_df1['num_morphemes'] - gold_df1['predicted']

std_residual = gold_df1['residual'].std()
filtered_df = gold_df1[np.abs(gold_df1['residual']) <= std_residual]
print(f"cleaned size (outliers removed): {filtered_df.shape}")
print(f"outliers removed: {len(gold_df1) - len(filtered_df):,} examples")

X_filtered = filtered_df[['word_len']]
y_filtered = filtered_df['num_morphemes']

model_filtered = LinearRegression()
model_filtered.fit(X_filtered, y_filtered)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=filtered_df, x='word_len', y='num_morphemes', alpha=0.5, label='Filtered Data')
plt.plot(X_filtered, model_filtered.predict(X_filtered), color='red', linewidth=2, label='Regression Line')
plt.title('Optimized Linear Regression: Word Length vs Morpheme Count')
plt.xlabel('Word Length (Characters)')
plt.ylabel('Number of Morphemes')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

slope = model.coef_[0]
intercept = model.intercept_
print(f"pre-refined regression: num_morphemes ≈ {slope:.2f} × word_len + {intercept:.2f}")

slope = model_filtered.coef_[0]
intercept = model_filtered.intercept_
print(f"refined regression: num_morphemes ≈ {slope:.2f} × word_len + {intercept:.2f}")

r2_full = r2_score(y, gold_df1['predicted'])
r2_filtered = r2_score(y_filtered, model_filtered.predict(X_filtered))
print(f"R2 (original): {r2_full:.3f}")
print(f"R2 (filtered): {r2_filtered:.3f}")'''
    nb.cells.append(new_code_cell(linear_regression))
    
    random_forest = '''# Random forest regression
gold_df2 = gold_df.copy()
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(gold_df2[['word_len']], gold_df2['num_morphemes'])

gold_df2['predicted_rf'] = rf.predict(gold_df2[['word_len']])
gold_df2['residual_rf'] = gold_df2['num_morphemes'] - gold_df2['predicted_rf']

mse_full = mean_squared_error(gold_df2['num_morphemes'], gold_df2['predicted_rf'])
mae_full = mean_absolute_error(gold_df2['num_morphemes'], gold_df2['predicted_rf'])
r2_full = r2_score(gold_df2['num_morphemes'], gold_df2['predicted_rf'])

std_residual = gold_df2['residual_rf'].std()
filtered_df_rf = gold_df2[np.abs(gold_df2['residual_rf']) <= std_residual].copy()

rf_filtered = RandomForestRegressor(n_estimators=100, random_state=42)
X_filtered = filtered_df_rf[['word_len']]
y_filtered = filtered_df_rf['num_morphemes']
rf_filtered.fit(X_filtered, y_filtered)

filtered_df_rf['predicted_rf'] = rf_filtered.predict(X_filtered)
r2_filtered = r2_score(y_filtered, filtered_df_rf['predicted_rf'])
mse_filtered = mean_squared_error(y_filtered, filtered_df_rf['predicted_rf'])
mae_filtered = mean_absolute_error(y_filtered, filtered_df_rf['predicted_rf'])

print("random forest (before outlier removal):")
print(f"MSE: {mse_full:.3f}")
print(f"MAE: {mae_full:.3f}")
print(f"R²:  {r2_full:.3f}")

print("random forest (after outlier removal):")
print(f"MSE: {mse_filtered:.3f}")
print(f"MAE: {mae_filtered:.3f}")
print(f"R²:  {r2_filtered:.3f}")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=filtered_df_rf['word_len'], y=filtered_df_rf['num_morphemes'], alpha=0.5, label='Filtered Data')
sns.lineplot(x=filtered_df_rf['word_len'], y=filtered_df_rf['predicted_rf'], color='red', label='RF Prediction (Filtered)')
plt.xlabel('Word Length (Characters)')
plt.ylabel('Number of Morphemes')
plt.title('Random Forest Regression (Filtered)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()'''
    nb.cells.append(new_code_cell(random_forest))
    
    polynomial = '''# Polynomial regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(gold_df2[['word_len']])
model_poly = LinearRegression().fit(X_poly, gold_df2['num_morphemes'])
preds_poly = model_poly.predict(X_poly)
r2_before = r2_score(gold_df2['num_morphemes'], preds_poly)
print(f"polynomial regression R2 (before filtering): {r2_before:.3f}")

residuals_poly = gold_df2['num_morphemes'] - preds_poly
std_resid_poly = residuals_poly.std()
mask = np.abs(residuals_poly) <= std_resid_poly
filtered_df_poly = gold_df2[mask].copy()

X_filtered_poly = poly.fit_transform(filtered_df_poly[['word_len']])
model_poly_filtered = LinearRegression().fit(X_filtered_poly, filtered_df_poly['num_morphemes'])
preds_filtered = model_poly_filtered.predict(X_filtered_poly)
r2_after = r2_score(filtered_df_poly['num_morphemes'], preds_filtered)
print(f"polynomial regression R2 (after filtering):  {r2_after:.3f}")'''
    nb.cells.append(new_code_cell(polynomial))
    
    outliers = '''# Identify outliers from all models
linear_outliers = gold_df1[np.abs(gold_df1['residual']) > std_residual]
rf_outliers = gold_df2[np.abs(gold_df2['residual_rf']) > std_residual]
poly_outliers = gold_df2[np.abs(residuals_poly) > std_resid_poly]

all_outliers = pd.concat([linear_outliers, rf_outliers, poly_outliers])
all_outliers = all_outliers[['word_len', 'num_morphemes']].drop_duplicates()

# Visualize outliers on heatmap
heatmap_data = gold_df.groupby(['word_len', 'num_morphemes']).size().unstack(fill_value=0)
outlier_coords = all_outliers[['word_len', 'num_morphemes']]
outlier_coords = outlier_coords[
    (outlier_coords['word_len'].isin(heatmap_data.index)) & 
    (outlier_coords['num_morphemes'].isin(heatmap_data.columns))
]
outlier_coords = outlier_coords.copy()
outlier_coords['freq'] = outlier_coords.apply(
    lambda row: heatmap_data.at[row['word_len'], row['num_morphemes']], axis=1
)

norm = plt.Normalize(vmin=heatmap_data.values.min(), vmax=heatmap_data.values.max())
colors = cm.Purples_r(norm(outlier_coords['freq'].values))

plt.figure(figsize=(10, 6))
ax = sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Reds', cbar_kws={'label': 'Frequency'})
plt.title('Heatmap of Word Length vs. Morpheme Count with Outliers')
plt.xlabel('Number of Morphemes')
plt.ylabel('Word Length (Characters)')

for j, (_, row) in enumerate(outlier_coords.iterrows()):
    plt.scatter(
        x=row['num_morphemes'] + 0.5,
        y=row['word_len'] + 0.5,
        color=colors[j],
        s=100,
        marker='X',
        linewidths=2,
        label='Outlier' if j == 0 else ""
    )

plt.tight_layout()
plt.show()'''
    nb.cells.append(new_code_cell(outliers))


def add_boundary_labels_section(nb):
    """Add boundary label generation function."""
    
    boundary_code = '''def get_boundary_labels(word, split):
    """Generate boundary labels for a word given its morpheme split."""
    labels = [0] * len(word)
    idx = 0
    for morpheme in split[:-1]:
        idx += len(morpheme)
        if idx < len(word):
            labels[idx - 1] = 1
    return labels'''
    nb.cells.append(new_code_cell(boundary_code))


def add_dataset_section(nb):
    """Add dataset class and model architecture."""
    
    dataset_code = '''class MorphemeDataset(Dataset):
    """Dataset for morphological segmentation training."""
    def __init__(self, path, char2idx=None):
        self.data = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if len(item["chars"]) == 0:
                    continue
                self.data.append(item)

        if char2idx is None:
            chars = set(c for item in self.data for c in item["chars"])
            self.char2idx = {c: i + 1 for i, c in enumerate(sorted(chars))}
            self.char2idx["<PAD>"] = 0
        else:
            self.char2idx = char2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        x = torch.tensor([self.char2idx[c] for c in item["chars"]], dtype=torch.long)
        y = torch.tensor(item["labels"], dtype=torch.float)
        return x, y

def collate_fn(batch):
    """Collate function: pads sequences to the same length."""
    xs, ys = zip(*batch)
    lengths = [len(x) for x in xs]
    max_len = max(lengths)
    padded_x = torch.zeros(len(xs), max_len, dtype=torch.long)
    padded_y = torch.zeros(len(ys), max_len, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        padded_x[i, :len(x)] = x
        padded_y[i, :len(y)] = y.long()
    mask = (padded_x != 0)
    return padded_x, padded_y, torch.tensor(lengths), mask

class Segmenter(nn.Module):
    """BiLSTM+CRF model for morphological segmentation."""
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=128, num_labels=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, x, lengths, labels=None, mask=None):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        h = self.relu(self.fc1(self.dropout(lstm_out)))
        emissions = self.fc2(h)

        if labels is not None:
            loss = -self.crf(emissions, labels.long(), mask=mask.bool(), reduction='mean')
            return loss
        else:
            best_paths = self.crf.decode(emissions, mask=mask.bool())
            return best_paths

def train(model, dataloader, epochs=10, lr=1e-3, device=None):
    """Training function for the Segmenter model."""
    if device is None:
        device = next(model.parameters()).device
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        for x, y, lengths, mask in dataloader:
            x = x.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            loss = model(x, lengths, labels=y, mask=mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch {epoch+1}: loss = {total_loss:.4f}")'''
    nb.cells.append(new_code_cell(dataset_code))


def add_checkpointing_section(nb):
    """Add model checkpointing functions."""
    
    checkpoint_code = '''def generate_model_id(emb_dim, hidden_dim, epochs, batch_size, lr, vocab_size, model_type="full"):
    """Hash training params to get unique model ID."""
    params_dict = {
        'emb_dim': emb_dim,
        'hidden_dim': hidden_dim,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'vocab_size': vocab_size,
        'model_type': model_type
    }
    params_str = json.dumps(params_dict, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()[:16]

def save_model_checkpoint(model, char2idx, model_id, models_folder=MODELS_FOLDER):
    """Save model checkpoint."""
    model_dir = os.path.join(models_folder, model_id)
    os.makedirs(model_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(model_dir, "segmenter_model.pt")
    torch.save({
        "model_state": model.state_dict(),
        "char2idx": char2idx
    }, checkpoint_path)
    
    metadata_path = os.path.join(model_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({
            'model_id': model_id,
            'vocab_size': len(char2idx),
            'model_name': MODEL_NAME
        }, f, indent=2)
    
    print(f"saved checkpoint to {model_dir}")
    return model_dir

def load_model_checkpoint(model_id, models_folder=MODELS_FOLDER):
    """Load model checkpoint."""
    model_dir = os.path.join(models_folder, model_id)
    checkpoint_path = os.path.join(model_dir, "segmenter_model.pt")
    
    if not os.path.exists(checkpoint_path):
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"loaded checkpoint from {model_dir}")
    return {
        'model_state': checkpoint['model_state'],
        'char2idx': checkpoint['char2idx'],
        'checkpoint_path': checkpoint_path,
        'model_dir': model_dir
    }'''
    nb.cells.append(new_code_cell(checkpoint_code))


def add_model1_section(nb):
    """Add Model 1 (full dataset) preparation and training."""
    
    prep_full = '''# Prepare full dataset for training
gold_df['char_seq'] = gold_df['Word'].apply(list)
gold_df['boundary_labels'] = gold_df.apply(
    lambda row: get_boundary_labels(row['Word'], row['Morph_split']), axis=1
)

# Save full dataset to JSONL
output_path = os.path.join(DATA_FOLDER, "stats_segmentation_data_full.jsonl")
with open(output_path, "w", encoding="utf-8") as f:
    for _, row in gold_df.iterrows():
        record = {
            "chars": row['char_seq'],
            "labels": row['boundary_labels']
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\\n")

print(f"full dataset saved to {output_path}")
print(f"  total examples: {len(gold_df):,}")'''
    nb.cells.append(new_code_cell(prep_full))
    
    train_model1 = '''# Model hyperparameters
EMB_DIM = 64
HIDDEN_DIM = 128
EPOCHS = 15
BATCH_SIZE = 16
LR = 3e-3

# Model 1: Trained on FULL dataset (with outliers)
data_path = os.path.join(DATA_FOLDER, "stats_segmentation_data_full.jsonl")
dataset = MorphemeDataset(data_path)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
vocab_size = len(dataset.char2idx)

model_id_full = generate_model_id(EMB_DIM, HIDDEN_DIM, EPOCHS, BATCH_SIZE, LR, vocab_size, model_type="full")

print(f"\\n=== MODEL 1: FULL DATASET (with outliers) ===")
print(f"looking for model {model_id_full}...")
loaded_full = load_model_checkpoint(model_id_full, models_folder=MODELS_FOLDER)

if loaded_full is not None:
    print(f"found it! loading from {loaded_full['model_dir']}")
    char2idx_full = loaded_full['char2idx']
    model_full = Segmenter(vocab_size=len(char2idx_full), emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM).to(device)
    model_full.load_state_dict(loaded_full['model_state'])
    model_full.eval()
    print("skipping training, model ready")
else:
    print(f"not found, training from scratch...")
    model_full = Segmenter(vocab_size, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM).to(device)
    train(model_full, dataloader, epochs=EPOCHS, lr=LR)
    save_model_checkpoint(model_full, dataset.char2idx, model_id_full, models_folder=MODELS_FOLDER)
    char2idx_full = dataset.char2idx
    print(f"\\nmodel 1 training done! model saved with ID: {model_id_full}")'''
    nb.cells.append(new_code_cell(train_model1))
    
    inference = '''def predict_segments(word, model, char2idx):
    """Predict morphological segmentation for a word."""
    model.eval()
    x = torch.tensor([[char2idx.get(c, 0) for c in word]], dtype=torch.long).to(device)
    lengths = torch.tensor([len(word)]).to(device)
    mask = (x != 0).to(device)
    with torch.no_grad():
        label_seq = model(x, lengths, labels=None, mask=mask)[0]
    
    segments = []
    start = 0
    for i, label in enumerate(label_seq):
        if label == 1:
            segments.append(word[start:i+1])
            start = i + 1
    if start < len(word):
        segments.append(word[start:])
    return segments

# Test Model 1
print("model 1 (full dataset) example:")
print(f"  pikunas -> {predict_segments('pikunas', model_full, char2idx_full)}")'''
    nb.cells.append(new_code_cell(inference))


def add_model2_section(nb):
    """Add Model 2 (filtered dataset) preparation and training."""
    
    prep_filtered = '''# Outlier detection and removal for Model 2
gold_df3 = gold_df.copy()
print(f"original size: {gold_df3.shape}")

X = gold_df3[['word_len']]
y = gold_df3['num_morphemes']

model = LinearRegression()
model.fit(X, y)
gold_df3['predicted'] = model.predict(X)
gold_df3['residual'] = gold_df3['num_morphemes'] - gold_df3['predicted']

std_residual = gold_df3['residual'].std()
filtered_df = gold_df3[np.abs(gold_df3['residual']) <= std_residual].copy()
print(f"cleaned size (outliers removed): {filtered_df.shape}")
print(f"outliers removed: {len(gold_df3) - len(filtered_df):,} examples ({100*(len(gold_df3) - len(filtered_df))/len(gold_df3):.1f}% of data)")

# Prepare filtered dataset
filtered_df = filtered_df.copy()
filtered_df['char_seq'] = filtered_df['Word'].apply(list)
filtered_df['boundary_labels'] = filtered_df.apply(
    lambda row: get_boundary_labels(row['Word'], row['Morph_split']), axis=1
)

# Save filtered dataset
output_path_filtered = os.path.join(DATA_FOLDER, "stats_segmentation_data_filtered.jsonl")
with open(output_path_filtered, "w", encoding="utf-8") as f:
    for _, row in filtered_df.iterrows():
        json.dump({
            "chars": row["char_seq"],
            "labels": row["boundary_labels"]
        }, f, ensure_ascii=False)
        f.write("\\n")

print(f"filtered dataset saved to {output_path_filtered}")
print(f"  total examples: {len(filtered_df):,}")'''
    nb.cells.append(new_code_cell(prep_filtered))
    
    train_model2 = '''# Model 2: Trained on FILTERED dataset (outliers removed)
data_path_filtered = os.path.join(DATA_FOLDER, "stats_segmentation_data_filtered.jsonl")
dataset2 = MorphemeDataset(data_path_filtered)
dataloader2 = DataLoader(dataset2, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
vocab_size_filtered = len(dataset2.char2idx)

model_id_filtered = generate_model_id(EMB_DIM, HIDDEN_DIM, EPOCHS, BATCH_SIZE, LR, vocab_size_filtered, model_type="filtered")

print(f"\\n=== MODEL 2: FILTERED DATASET (outliers removed) ===")
print(f"looking for model {model_id_filtered}...")
loaded_filtered = load_model_checkpoint(model_id_filtered, models_folder=MODELS_FOLDER)

if loaded_filtered is not None:
    print(f"found it! loading from {loaded_filtered['model_dir']}")
    char2idx_filtered = loaded_filtered['char2idx']
    model_filtered = Segmenter(vocab_size=len(char2idx_filtered), emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM).to(device)
    model_filtered.load_state_dict(loaded_filtered['model_state'])
    model_filtered.eval()
    print("skipping training, model ready")
else:
    print(f"not found, training from scratch...")
    model_filtered = Segmenter(vocab_size_filtered, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM).to(device)
    train(model_filtered, dataloader2, epochs=EPOCHS, lr=LR)
    save_model_checkpoint(model_filtered, dataset2.char2idx, model_id_filtered, models_folder=MODELS_FOLDER)
    char2idx_filtered = dataset2.char2idx
    print(f"\\nmodel 2 training done! model saved with ID: {model_id_filtered}")

# Test Model 2
print("model 2 (filtered dataset) example:")
print(f"  pikunas -> {predict_segments('pikunas', model_filtered, char2idx_filtered)}")'''
    nb.cells.append(new_code_cell(train_model2))


def add_kfold_section(nb):
    """Add k-fold cross-validation function."""
    
    kfold_code = '''def run_kfold_cross_validation(
    df,
    n_folds=5,
    emb_dim=64,
    hidden_dim=128,
    epochs=15,
    batch_size=16,
    lr=3e-3,
    random_state=42,
    device=device
):
    """K-fold cross-validation for more robust evaluation."""
    print(f"\\n{'=' * 80}")
    print(f"K-FOLD CV (k={n_folds}) WITH BILSTM+CRF")
    print(f"{'=' * 80}")
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    indices = np.arange(len(df))
    
    fold_results = []
    all_metrics = {
        'val_loss': [],
        'exact_match': []
    }
    
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(indices), 1):
        print(f"\\n--- fold {fold_idx}/{n_folds} ---")
        print(f"train: {len(train_indices)}, val: {len(val_indices)}")
        
        train_df_fold = df.iloc[train_indices].reset_index(drop=True)
        val_df_fold = df.iloc[val_indices].reset_index(drop=True)
        
        import tempfile
        import os as os_module
        
        temp_dir = tempfile.mkdtemp()
        train_path_fold = os_module.path.join(temp_dir, f"train_fold_{fold_idx}.jsonl")
        val_path_fold = os_module.path.join(temp_dir, f"val_fold_{fold_idx}.jsonl")
        
        with open(train_path_fold, "w", encoding="utf-8") as f:
            for _, row in train_df_fold.iterrows():
                record = {
                    "chars": row['char_seq'],
                    "labels": row['boundary_labels']
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\\n")
        
        with open(val_path_fold, "w", encoding="utf-8") as f:
            for _, row in val_df_fold.iterrows():
                record = {
                    "chars": row['char_seq'],
                    "labels": row['boundary_labels']
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\\n")
        
        train_dataset_fold = MorphemeDataset(train_path_fold)
        val_dataset_fold = MorphemeDataset(val_path_fold, char2idx=train_dataset_fold.char2idx)
        train_loader_fold = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader_fold = DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        vocab_size_fold = len(train_dataset_fold.char2idx)
        
        model_fold = Segmenter(vocab_size=vocab_size_fold, emb_dim=emb_dim, hidden_dim=hidden_dim).to(device)
        
        best_val_loss = float('inf')
        best_epoch = 0
        best_exact_match = 0.0
        
        optimizer_fold = torch.optim.Adam(model_fold.parameters(), lr=lr)
        
        for epoch in range(1, epochs + 1):
            model_fold.train()
            total_loss = 0.0
            for x, y, lengths, mask in train_loader_fold:
                x = x.to(device)
                y = y.to(device)
                lengths = lengths.to(device)
                mask = mask.to(device)
                
                optimizer_fold.zero_grad()
                loss = model_fold(x, lengths, labels=y, mask=mask)
                loss.backward()
                optimizer_fold.step()
                
                total_loss += loss.item()
            
            train_loss = total_loss / len(train_loader_fold)
            
            model_fold.eval()
            val_loss = 0.0
            exact_matches = 0
            total_val = 0
            
            def predict_segments_fold(word, model, char2idx):
                """Fold-specific version of predict_segments."""
                model.eval()
                x = torch.tensor([[char2idx.get(c, 0) for c in word]], dtype=torch.long).to(device)
                lengths = torch.tensor([len(word)]).to(device)
                mask = (x != 0).to(device)
                with torch.no_grad():
                    label_seq = model(x, lengths, labels=None, mask=mask)[0]
                segments = []
                start = 0
                for i, label in enumerate(label_seq):
                    if label == 1:
                        segments.append(word[start:i+1])
                        start = i + 1
                if start < len(word):
                    segments.append(word[start:])
                return segments
            
            with torch.no_grad():
                for x, y, lengths, mask in val_loader_fold:
                    x = x.to(device)
                    y = y.to(device)
                    lengths = lengths.to(device)
                    mask = mask.to(device)
                    
                    loss = model_fold(x, lengths, labels=y, mask=mask)
                    val_loss += loss.item()
                
                for i in range(len(val_dataset_fold)):
                    word = val_df_fold.iloc[i]['Word']
                    gold_split = val_df_fold.iloc[i]['Morph_split']
                    
                    predicted_segments = predict_segments_fold(word, model_fold, train_dataset_fold.char2idx)
                    
                    if predicted_segments == gold_split:
                        exact_matches += 1
                    total_val += 1
            
            val_loss = val_loss / len(val_loader_fold)
            exact_match_rate = exact_matches / total_val if total_val > 0 else 0.0
            
            print(f"  ep {epoch:02d} | train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  exact_match={exact_match_rate:.3f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_exact_match = exact_match_rate
                best_epoch = epoch
        
        print(f"\\n  best epoch: {best_epoch}")
        print(f"  best validation: loss={best_val_loss:.4f}  exact_match={best_exact_match:.3f}")
        
        try:
            os_module.remove(train_path_fold)
            os_module.remove(val_path_fold)
            os_module.rmdir(temp_dir)
        except:
            pass
        
        fold_results.append({
            'fold': fold_idx,
            'val_loss': best_val_loss,
            'exact_match': best_exact_match,
            'best_epoch': best_epoch
        })
        
        all_metrics['val_loss'].append(best_val_loss)
        all_metrics['exact_match'].append(best_exact_match)
    
    mean_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    std_metrics = {k: np.std(v) for k, v in all_metrics.items()}
    best_fold_idx = min(range(len(fold_results)), key=lambda i: fold_results[i]['val_loss'])
    
    print(f"\\n{'=' * 80}")
    print("CV SUMMARY")
    print(f"{'=' * 80}")
    for r in fold_results:
        print(f"  fold {r['fold']}: loss={r['val_loss']:.4f}, exact_match={r['exact_match']:.3f}")
    
    print(f"\\nmean +/- std over {n_folds} folds:")
    print(f"  validation loss:   {mean_metrics['val_loss']:.4f} +/- {std_metrics['val_loss']:.4f}")
    print(f"  exact match rate:  {mean_metrics['exact_match']:.3f} +/- {std_metrics['exact_match']:.3f}")
    print(f"\\nbest fold: {fold_results[best_fold_idx]['fold']} "
          f"(loss: {fold_results[best_fold_idx]['val_loss']:.4f}, "
          f"exact_match: {fold_results[best_fold_idx]['exact_match']:.3f})")
    print(f"{'=' * 80}\\n")
    
    return {
        'fold_results': fold_results,
        'mean_metrics': mean_metrics,
        'std_metrics': std_metrics,
        'best_fold_idx': best_fold_idx,
        'all_metrics': all_metrics
    }'''
    nb.cells.append(new_code_cell(kfold_code))


def add_run_kfold_section(nb):
    """Add k-fold CV execution."""
    
    run_kfold = '''# Run k-fold cross-validation on full dataset
kfold_results_full = run_kfold_cross_validation(
    df=gold_df,
    n_folds=5,
    emb_dim=EMB_DIM if 'EMB_DIM' in globals() else 64,
    hidden_dim=HIDDEN_DIM if 'HIDDEN_DIM' in globals() else 128,
    epochs=EPOCHS if 'EPOCHS' in globals() else 15,
    batch_size=BATCH_SIZE if 'BATCH_SIZE' in globals() else 16,
    lr=LR if 'LR' in globals() else 3e-3,
    random_state=42,
    device=device
)

print(f"\\navg exact match rate: {kfold_results_full['mean_metrics']['exact_match']:.3f} +/- {kfold_results_full['std_metrics']['exact_match']:.3f}")
print(f"avg validation loss: {kfold_results_full['mean_metrics']['val_loss']:.4f} +/- {kfold_results_full['std_metrics']['val_loss']:.4f}")'''
    nb.cells.append(new_code_cell(run_kfold))


def add_evaluation_section(nb):
    """Add evaluation functions and comparison."""
    
    eval_helpers = '''def is_correct_prediction(predicted, gold_variants):
    """Check if predicted segmentation matches any gold variant."""
    if gold_variants is None:
        return False
    
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
        gold_variants = normalized
    
    return any(predicted == variant for variant in gold_variants)

def split_count_metrics(predicted_segments, gold_variants):
    """Compute split-count accuracy variants."""
    pred_count = len(predicted_segments)
    
    if gold_variants is None:
        return {"Exact": False, "+1": False, "-1": False, "±1": False}
    
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
        gold_variants = normalized
    
    gold_counts = [len(gold) for gold in gold_variants]

    exact = any(pred_count == g for g in gold_counts)
    plus1 = any(pred_count == g + 1 for g in gold_counts)
    minus1 = any(pred_count == g - 1 for g in gold_counts)
    pm1 = any(abs(pred_count - g) <= 1 for g in gold_counts)

    return {"Exact": exact, "+1": plus1, "-1": minus1, "±1": pm1}'''
    nb.cells.append(new_code_cell(eval_helpers))
    
    test_eval = '''# Load test data
print("loading test data...")
test_df = pd.read_parquet(os.path.join(DATA_FOLDER, "cleaned_data_df.parquet"))
print(f"loaded {len(test_df):,} test examples")

# Evaluate both models
records_full = []
records_filtered = []
all_words = test_df["Word"].tolist()

print("\\nevaluating model 1 (full dataset with outliers)...")
for word in all_words:
    predicted_full = predict_segments(word, model_full, char2idx_full)
    gold_variants = test_df[test_df["Word"] == word]["Gold"].iloc[0] if len(test_df[test_df["Word"] == word]) > 0 else []
    
    correct_exact_full = is_correct_prediction(predicted_full, gold_variants)
    split_metrics_full = split_count_metrics(predicted_full, gold_variants)
    
    records_full.append({
        "Word": word,
        "Prediction": predicted_full,
        "Gold": gold_variants,
        "CorrectExactSeg": correct_exact_full,
        "CorrectSplitCount": split_metrics_full["Exact"],
        "SplitCount+1": split_metrics_full["+1"],
        "SplitCount-1": split_metrics_full["-1"],
        "SplitCount±1": split_metrics_full["±1"],
        "OverlapExactAndSplit": correct_exact_full and split_metrics_full["Exact"]
    })

print("evaluating model 2 (filtered dataset without outliers)...")
for word in all_words:
    predicted_filtered = predict_segments(word, model_filtered, char2idx_filtered)
    gold_variants = test_df[test_df["Word"] == word]["Gold"].iloc[0] if len(test_df[test_df["Word"] == word]) > 0 else []
    
    correct_exact_filtered = is_correct_prediction(predicted_filtered, gold_variants)
    split_metrics_filtered = split_count_metrics(predicted_filtered, gold_variants)
    
    records_filtered.append({
        "Word": word,
        "Prediction": predicted_filtered,
        "Gold": gold_variants,
        "CorrectExactSeg": correct_exact_filtered,
        "CorrectSplitCount": split_metrics_filtered["Exact"],
        "SplitCount+1": split_metrics_filtered["+1"],
        "SplitCount-1": split_metrics_filtered["-1"],
        "SplitCount±1": split_metrics_filtered["±1"],
        "OverlapExactAndSplit": correct_exact_filtered and split_metrics_filtered["Exact"]
    })

results_full_df = pd.DataFrame(records_full)
results_filtered_df = pd.DataFrame(records_filtered)

# Compute aggregate metrics
accuracy_full = results_full_df["CorrectExactSeg"].mean()
split_exact_full = results_full_df["CorrectSplitCount"].mean()
split_plus1_full = results_full_df["SplitCount+1"].mean()
split_minus1_full = results_full_df["SplitCount-1"].mean()
split_pm1_full = results_full_df["SplitCount±1"].mean()
overlap_full = results_full_df["OverlapExactAndSplit"].mean()

accuracy_filtered = results_filtered_df["CorrectExactSeg"].mean()
split_exact_filtered = results_filtered_df["CorrectSplitCount"].mean()
split_plus1_filtered = results_filtered_df["SplitCount+1"].mean()
split_minus1_filtered = results_filtered_df["SplitCount-1"].mean()
split_pm1_filtered = results_filtered_df["SplitCount±1"].mean()
overlap_filtered = results_filtered_df["OverlapExactAndSplit"].mean()

# Comparison results
print("\\n" + "="*80)
print("comparing models: full dataset vs filtered dataset (outliers removed)")
print("="*80)
print("\\nchecking if removing outliers helps or hurts")
print(f"outliers removed: {len(gold_df) - len(filtered_df):,} examples ({100*(len(gold_df) - len(filtered_df))/len(gold_df):.1f}% of data)")
print("\\n" + "-"*80)
print(f"{'Metric':<30} {'Model 1 (Full)':<20} {'Model 2 (Filtered)':<20} {'Difference':<15}")
print("-"*80)
print(f"{'Exact Segmentation Accuracy':<30} {accuracy_full:<20.4f} {accuracy_filtered:<20.4f} {accuracy_filtered-accuracy_full:+.4f}")
print(f"{'Split-count (Exact)':<30} {split_exact_full:<20.4f} {split_exact_filtered:<20.4f} {split_exact_filtered-split_exact_full:+.4f}")
print(f"{'Split-count (+1)':<30} {split_plus1_full:<20.4f} {split_plus1_filtered:<20.4f} {split_plus1_filtered-split_plus1_full:+.4f}")
print(f"{'Split-count (−1)':<30} {split_minus1_full:<20.4f} {split_minus1_filtered:<20.4f} {split_minus1_filtered-split_minus1_full:+.4f}")
print(f"{'Split-count (±1)':<30} {split_pm1_full:<20.4f} {split_pm1_filtered:<20.4f} {split_pm1_filtered-split_pm1_full:+.4f}")
print(f"{'Overlap (Exact ∩ Split)':<30} {overlap_full:<20.4f} {overlap_filtered:<20.4f} {overlap_filtered-overlap_full:+.4f}")
print("-"*80)

if accuracy_filtered > accuracy_full:
    print(f"\\nmodel 2 (filtered) did better than model 1 (full)")
    print(f"  went up by {100*(accuracy_filtered-accuracy_full):.2f} percentage points")
    print("  so removing outliers seems to help")
elif accuracy_filtered < accuracy_full:
    print(f"\\nmodel 2 (filtered) did worse than model 1 (full)")
    print(f"  went down by {100*(accuracy_full-accuracy_filtered):.2f} percentage points")
    print("  so removing outliers seems to hurt")
else:
    print(f"\\nmodel 2 (filtered) did the same as model 1 (full)")
    print("  so removing outliers doesn't seem to matter")

# Save results
results_full_path = os.path.join(DATA_FOLDER, "stats_model_full_eval_results.csv")
results_filtered_path = os.path.join(DATA_FOLDER, "stats_model_filtered_eval_results.csv")

results_full_df.to_csv(results_full_path, index=False)
results_filtered_df.to_csv(results_filtered_path, index=False)

print(f"\\nevaluation results saved:")
print(f"  model 1 (full): {results_full_path}")
print(f"  model 2 (filtered): {results_filtered_path}")'''
    nb.cells.append(new_code_cell(test_eval))


def main():
    """Build and save the refactored notebook."""
    nb = create_notebook()
    
    # Add sections in order
    add_header_section(nb)
    add_imports_section(nb)
    add_config_section(nb)
    add_data_loading_section(nb)
    add_statistical_analysis_section(nb)
    add_boundary_labels_section(nb)
    add_dataset_section(nb)
    add_checkpointing_section(nb)
    add_model1_section(nb)
    add_model2_section(nb)
    add_kfold_section(nb)
    add_run_kfold_section(nb)
    add_evaluation_section(nb)
    
    # Save notebook
    output_path = "stats_refactored.ipynb"
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print(f"Refactored notebook saved to: {output_path}")


if __name__ == "__main__":
    main()

