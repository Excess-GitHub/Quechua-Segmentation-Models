#!/usr/bin/env python3
"""
Script to generate a refactored version of the transmorpher notebook.
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
        "# Transmorpher: Transformer-Based Morphology Parser\n\n"
        "Transformer encoder-decoder for Quechua morphological segmentation. "
        "Generates segmented words directly (e.g., 'pikunas' -> 'pi+kuna+s') "
        "using sequence-to-sequence architecture."
    ))


def add_imports_section(nb):
    """Add all imports in a single organized cell."""
    imports_code = '''# Core libraries
import os
import ast
import json
import hashlib
import pickle
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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader'''
    nb.cells.append(new_code_cell(imports_code))


def add_config_section(nb):
    """Add configuration constants."""
    config_code = '''# Paths
DATA_FOLDER = "data"
MODEL_NAME = "transmorpher"
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
gold_df.drop_duplicates(subset='Word', keep='first', inplace=True)
gold_df.dropna(subset=['Word'], inplace=True)
print(f"got {len(gold_df):,} gold examples")'''
    nb.cells.append(new_code_cell(data_loading))
    
    features = '''# Extract basic features
gold_df['num_morphemes'] = gold_df['Morph_split'].apply(len)
gold_df['word_len'] = gold_df['Word'].apply(len)'''
    nb.cells.append(new_code_cell(features))
    
    segmentation = '''# Convert morpheme splits to segmentation format with '+' separators
gold_df['segmentation'] = gold_df['Morph_split_str'].str.replace(' ', '+')'''
    nb.cells.append(new_code_cell(segmentation))


def add_data_analysis_section(nb):
    """Add data analysis section (correlation, regression, outliers)."""
    
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
print(f"spearman correlation: {spearman_corr:.3f} (p={spearman_p:.3e})")'''
    nb.cells.append(new_code_cell(correlation))
    
    linear_regression = '''# Linear regression with outlier removal
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
print(f"cleaned size: {filtered_df.shape}")

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


def add_vocabulary_section(nb):
    """Add vocabulary construction."""
    
    vocab_code = '''# Build character-level vocabularies
special_tokens = ['<pad>', '<s>', '</s>']
input_chars = sorted({ch for word in gold_df['Word'] for ch in word})
output_chars = sorted({ch for seg in gold_df['segmentation'] for ch in seg})
input_vocab = special_tokens + input_chars
output_vocab = special_tokens + output_chars
input2idx = {ch: idx for idx, ch in enumerate(input_vocab)}
output2idx = {ch: idx for idx, ch in enumerate(output_vocab)}
PAD_IN, START_IN, END_IN = input2idx['<pad>'], input2idx['<s>'], input2idx['</s>']
PAD_OUT, START_OUT, END_OUT = output2idx['<pad>'], output2idx['<s>'], output2idx['</s>']'''
    nb.cells.append(new_code_cell(vocab_code))


def add_dataset_section(nb):
    """Add dataset class and dataloaders."""
    
    dataset_code = '''class QuechuaSegDataset(Dataset):
    """Dataset for Quechua morphological segmentation."""
    def __init__(self, df, input2idx, output2idx, max_input_len=None, max_output_len=None):
        self.words = df['Word'].tolist()
        self.segs = df['segmentation'].tolist()
        self.input2idx = input2idx
        self.output2idx = output2idx
        self.max_input_len = max_input_len or max(len(w) for w in self.words) + 2
        self.max_output_len = max_output_len or max(len(s) for s in self.segs) + 2

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        seg = self.segs[idx]
        src = [self.input2idx.get(ch, PAD_IN) for ch in word]
        src = [START_IN] + src + [END_IN]
        src += [PAD_IN] * (self.max_input_len - len(src))
        tgt = [START_OUT] + [self.output2idx[ch] for ch in seg] + [END_OUT]
        tgt += [PAD_OUT] * (self.max_output_len - len(tgt))
        return torch.tensor(src), torch.tensor(tgt)'''
    nb.cells.append(new_code_cell(dataset_code))
    
    split_code = '''# Train/validation split
dataset = QuechuaSegDataset(gold_df, input2idx, output2idx)
n_train = int(0.8 * len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
print(f"training: {len(train_dataset):,} samples")
print(f"validation: {len(val_dataset):,} samples")'''
    nb.cells.append(new_code_cell(split_code))


def add_model_section(nb):
    """Add Transformer model definition."""
    
    model_code = '''class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer models."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0)]

class MorphSegModel(nn.Module):
    """Transformer encoder-decoder for morphological segmentation."""
    def __init__(self, in_vocab, out_vocab, d_model=64, ff=128, heads=2, layers=1, drop=0.0):
        super().__init__()
        self.enc_embed = nn.Embedding(in_vocab, d_model, padding_idx=PAD_IN)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, heads, ff, drop, batch_first=False)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)

        self.dec_embed = nn.Embedding(out_vocab, d_model, padding_idx=PAD_OUT)
        self.pos_dec = PositionalEncoding(d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model, heads, ff, drop, batch_first=False)
        self.decoder = nn.TransformerDecoder(dec_layer, layers)

        self.out_proj = nn.Linear(d_model, out_vocab)

    def forward(self, src, tgt, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        e_src = self.pos_enc(self.enc_embed(src))
        memory = self.encoder(e_src, src_key_padding_mask=src_key_padding_mask)
        
        e_tgt = self.pos_dec(self.dec_embed(tgt))
        out = self.decoder(e_tgt, memory, tgt_mask=tgt_mask,
                           memory_key_padding_mask=src_key_padding_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask)
        
        return self.out_proj(out)'''
    nb.cells.append(new_code_cell(model_code))


def add_checkpointing_section(nb):
    """Add model checkpointing functions."""
    
    checkpoint_code = '''def generate_model_id(d_model, ff, heads, layers, drop, epochs, batch_size, lr, in_vocab_size, out_vocab_size):
    """Hash training params to get unique model ID."""
    params_dict = {
        'd_model': d_model,
        'ff': ff,
        'heads': heads,
        'layers': layers,
        'drop': drop,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'in_vocab_size': in_vocab_size,
        'out_vocab_size': out_vocab_size
    }
    params_str = json.dumps(params_dict, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()[:16]

def save_model_checkpoint(model, input2idx, output2idx, model_id, models_folder=MODELS_FOLDER):
    """Save model checkpoint."""
    model_dir = os.path.join(models_folder, model_id)
    os.makedirs(model_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(model_dir, "transformer_morph_seg.pt")
    torch.save({
        "model_state": model.state_dict(),
        "input2idx": input2idx,
        "output2idx": output2idx
    }, checkpoint_path)
    
    metadata_path = os.path.join(model_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({
            'model_id': model_id,
            'in_vocab_size': len(input2idx),
            'out_vocab_size': len(output2idx),
            'model_name': MODEL_NAME
        }, f, indent=2)
    
    print(f"saved checkpoint to {model_dir}")
    return model_dir

def load_model_checkpoint(model_id, models_folder=MODELS_FOLDER):
    """Load model checkpoint."""
    model_dir = os.path.join(models_folder, model_id)
    checkpoint_path = os.path.join(model_dir, "transformer_morph_seg.pt")
    
    if not os.path.exists(checkpoint_path):
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"loaded checkpoint from {model_dir}")
    return {
        'model_state': checkpoint['model_state'],
        'input2idx': checkpoint['input2idx'],
        'output2idx': checkpoint['output2idx'],
        'checkpoint_path': checkpoint_path,
        'model_dir': model_dir
    }'''
    nb.cells.append(new_code_cell(checkpoint_code))


def add_training_section(nb):
    """Add training loop."""
    
    training_code = '''# Model hyperparameters
D_MODEL = 64
FF = 128
HEADS = 2
LAYERS = 1
DROP = 0.0
EPOCHS = 15
BATCH_SIZE = 16
LR = 1e-4

# Generate model identifier
model_id = generate_model_id(D_MODEL, FF, HEADS, LAYERS, DROP, EPOCHS, BATCH_SIZE, LR, 
                              len(input_vocab), len(output_vocab))

# Try to load existing model
print(f"looking for model {model_id}...")
loaded = load_model_checkpoint(model_id, models_folder=MODELS_FOLDER)

if loaded is not None:
    print(f"found it! loading from {loaded['model_dir']}")
    input2idx = loaded['input2idx']
    output2idx = loaded['output2idx']
    PAD_IN, START_IN, END_IN = input2idx['<pad>'], input2idx['<s>'], input2idx['</s>']
    PAD_OUT, START_OUT, END_OUT = output2idx['<pad>'], output2idx['<s>'], output2idx['</s>']
    model = MorphSegModel(len(input2idx), len(output2idx), d_model=D_MODEL, ff=FF, 
                          heads=HEADS, layers=LAYERS, drop=DROP).to(device)
    model.load_state_dict(loaded['model_state'])
    model.eval()
    print("skipping training, model ready")
else:
    print(f"not found, training from scratch...")
    model = MorphSegModel(len(input_vocab), len(output_vocab), d_model=D_MODEL, ff=FF, 
                          heads=HEADS, layers=LAYERS, drop=DROP).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_OUT)
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for src_batch, tgt_batch in train_loader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            src_pad_mask = (src_batch == PAD_IN)
            tgt_pad_mask = (tgt_batch == PAD_OUT)

            src = src_batch.transpose(0, 1)
            tgt = tgt_batch.transpose(0, 1)
            tgt_input = tgt[:-1, :]
            tgt_output = tgt[1:, :]

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                           tgt_input.size(0)
                       ).to(device)

            logits = model(
                src,
                tgt_input,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_pad_mask,
                tgt_key_padding_mask=tgt_pad_mask[:, :-1]
            )
            
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"epoch {epoch:02d} — train loss: {avg_loss:.4f}")
    
    save_model_checkpoint(model, input2idx, output2idx, model_id, models_folder=MODELS_FOLDER)
    print(f"\\ntraining done! model saved with ID: {model_id}")'''
    nb.cells.append(new_code_cell(training_code))


def add_inference_section(nb):
    """Add inference function."""
    
    inference_code = '''def segment_word(word, model, in2idx, out2idx, idx2out, max_len=50, debug=False):
    """Segment a word using the trained Transformer model."""
    model.eval()
    src_idx = [START_IN] + [in2idx.get(ch, PAD_IN) for ch in word] + [END_IN]
    src = torch.tensor(src_idx).unsqueeze(1).to(device)
    src_pad = (src.squeeze(1) == PAD_IN).unsqueeze(0)
    
    with torch.no_grad():
        mem = model.pos_enc(model.enc_embed(src))
        mem = model.encoder(mem, src_key_padding_mask=src_pad)
    
    out_ids = [START_OUT]
    if debug:
        print(f"segmenting '{word}':")
        print(f"  generated tokens so far: ", end="")
    
    for step in range(max_len):
        tgt = torch.tensor(out_ids).unsqueeze(1).to(device)
        mask = nn.Transformer.generate_square_subsequent_mask(len(out_ids)).to(device)
        with torch.no_grad():
            dec_out = model.pos_dec(model.dec_embed(tgt))
            dec = model.decoder(dec_out, mem, tgt_mask=mask, memory_key_padding_mask=src_pad)
            logits = model.out_proj(dec)
        
        probs = torch.softmax(logits[-1, 0], dim=0)
        top_probs, top_indices = torch.topk(probs, k=3)
        
        nxt = logits[-1, 0].argmax().item()
        
        if debug:
            top_chars = [idx2out[idx.item()] for idx in top_indices]
            print(f"\\n  step {step+1}: top predictions = {list(zip(top_chars, top_probs.tolist()))}")
            print(f"    -> selected: '{idx2out[nxt]}' (prob={probs[nxt].item():.4f})")
        
        current_output = ''.join(idx2out[i] for i in out_ids[1:])
        current_chars = len(current_output.replace('+', ''))
        input_chars = len(word)
        min_expected_chars = input_chars
        
        if nxt == END_OUT:
            end_prob = probs[END_OUT].item()
            if current_chars >= min_expected_chars or end_prob > 0.8:
                if debug:
                    print(f"  stopped at END_OUT token (chars: {current_chars}/{min_expected_chars}, prob: {end_prob:.4f})")
                break
            else:
                if debug:
                    print(f"  END_OUT predicted but too early (chars: {current_chars}/{min_expected_chars}), forcing continuation...")
                for idx in top_indices:
                    if idx.item() != END_OUT:
                        nxt = idx.item()
                        if debug:
                            print(f"    -> forced selection: '{idx2out[nxt]}' (prob={probs[nxt].item():.4f})")
                        break
        
        out_ids.append(nxt)
        
        if debug:
            current_seg = ''.join(idx2out[i] for i in out_ids[1:])
            print(f"    current segmentation: '{current_seg}'")
    
    result = ''.join(idx2out[i] for i in out_ids[1:])
    
    if debug:
        print(f"  final result: '{result}'")
        print(f"  expected length check: input '{word}' ({len(word)} chars) -> output '{result}' ({len(result.replace('+', ''))} chars)")
    
    return result

# Build reverse vocabulary mapping
idx2output = {idx: ch for ch, idx in output2idx.items()}'''
    nb.cells.append(new_code_cell(inference_code))


def add_kfold_section(nb):
    """Add k-fold cross-validation function."""
    
    kfold_code = '''def run_kfold_cross_validation(
    df,
    n_folds=5,
    d_model=64,
    ff=128,
    heads=2,
    layers=1,
    drop=0.0,
    epochs=15,
    batch_size=16,
    lr=1e-4,
    random_state=42,
    device=device
):
    """K-fold cross-validation for more robust evaluation."""
    print(f"\\n{'=' * 80}")
    print(f"K-FOLD CV (k={n_folds}) WITH TRANSFORMER")
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
        
        print(f"  building vocabularies from training fold...")
        special_tokens = ['<pad>', '<s>', '</s>']
        input_chars_fold = sorted({ch for word in train_df_fold['Word'] for ch in word})
        output_chars_fold = sorted({ch for seg in train_df_fold['segmentation'] for ch in seg})
        input_vocab_fold = special_tokens + input_chars_fold
        output_vocab_fold = special_tokens + output_chars_fold
        input2idx_fold = {ch: idx for idx, ch in enumerate(input_vocab_fold)}
        output2idx_fold = {ch: idx for idx, ch in enumerate(output_vocab_fold)}
        PAD_IN_FOLD = input2idx_fold['<pad>']
        START_IN_FOLD = input2idx_fold['<s>']
        END_IN_FOLD = input2idx_fold['</s>']
        PAD_OUT_FOLD = output2idx_fold['<pad>']
        START_OUT_FOLD = output2idx_fold['<s>']
        END_OUT_FOLD = output2idx_fold['</s>']
        
        train_dataset_fold = QuechuaSegDataset(train_df_fold, input2idx_fold, output2idx_fold)
        val_dataset_fold = QuechuaSegDataset(val_df_fold, input2idx_fold, output2idx_fold)
        train_loader_fold = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True)
        val_loader_fold = DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=False)
        
        model_fold = MorphSegModel(
            len(input_vocab_fold), 
            len(output_vocab_fold), 
            d_model=d_model, 
            ff=ff, 
            heads=heads, 
            layers=layers, 
            drop=drop
        ).to(device)
        
        criterion_fold = nn.CrossEntropyLoss(ignore_index=PAD_OUT_FOLD)
        optimizer_fold = optim.Adam(model_fold.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        
        best_val_loss = float('inf')
        best_epoch = 0
        best_exact_match = 0.0
        
        for epoch in range(1, epochs + 1):
            model_fold.train()
            total_loss = 0.0
            for src_batch, tgt_batch in train_loader_fold:
                src_batch = src_batch.to(device)
                tgt_batch = tgt_batch.to(device)
                
                src_pad_mask = (src_batch == PAD_IN_FOLD)
                tgt_pad_mask = (tgt_batch == PAD_OUT_FOLD)
                
                src = src_batch.transpose(0, 1)
                tgt = tgt_batch.transpose(0, 1)
                tgt_input = tgt[:-1, :]
                tgt_output = tgt[1:, :]
                
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    tgt_input.size(0)
                ).to(device)
                
                logits = model_fold(
                    src,
                    tgt_input,
                    tgt_mask=tgt_mask,
                    src_key_padding_mask=src_pad_mask,
                    tgt_key_padding_mask=tgt_pad_mask[:, :-1]
                )
                
                loss = criterion_fold(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_output.reshape(-1)
                )
                
                optimizer_fold.zero_grad()
                loss.backward()
                optimizer_fold.step()
                
                total_loss += loss.item()
            
            train_loss = total_loss / len(train_loader_fold)
            
            model_fold.eval()
            val_loss = 0.0
            exact_matches = 0
            total_val = 0
            
            idx2output_fold = {idx: ch for ch, idx in output2idx_fold.items()}
            
            def segment_word_fold(word, model, in2idx, out2idx, idx2out, max_len=50):
                """Fold-specific version of segment_word."""
                model.eval()
                src_idx = [START_IN_FOLD] + [in2idx.get(ch, PAD_IN_FOLD) for ch in word] + [END_IN_FOLD]
                src = torch.tensor(src_idx).unsqueeze(1).to(device)
                src_pad = (src.squeeze(1) == PAD_IN_FOLD).unsqueeze(0)
                
                with torch.no_grad():
                    mem = model.pos_enc(model.enc_embed(src))
                    mem = model.encoder(mem, src_key_padding_mask=src_pad)
                
                out_ids = [START_OUT_FOLD]
                for step in range(max_len):
                    tgt = torch.tensor(out_ids).unsqueeze(1).to(device)
                    mask = nn.Transformer.generate_square_subsequent_mask(len(out_ids)).to(device)
                    with torch.no_grad():
                        dec_out = model.pos_dec(model.dec_embed(tgt))
                        dec = model.decoder(dec_out, mem, tgt_mask=mask, memory_key_padding_mask=src_pad)
                        logits = model.out_proj(dec)
                    
                    probs = torch.softmax(logits[-1, 0], dim=0)
                    nxt = logits[-1, 0].argmax().item()
                    
                    current_output = ''.join(idx2out[i] for i in out_ids[1:])
                    current_chars = len(current_output.replace('+', ''))
                    input_chars = len(word)
                    min_expected_chars = input_chars
                    
                    if nxt == END_OUT_FOLD:
                        end_prob = probs[END_OUT_FOLD].item()
                        if current_chars >= min_expected_chars or end_prob > 0.8:
                            break
                        else:
                            top_probs, top_indices = torch.topk(probs, k=3)
                            for idx in top_indices:
                                if idx.item() != END_OUT_FOLD:
                                    nxt = idx.item()
                                    break
                    
                    out_ids.append(nxt)
                
                result = ''.join(idx2out[i] for i in out_ids[1:])
                return result
            
            with torch.no_grad():
                for src_batch, tgt_batch in val_loader_fold:
                    src_batch = src_batch.to(device)
                    tgt_batch = tgt_batch.to(device)
                    
                    src_pad_mask = (src_batch == PAD_IN_FOLD)
                    tgt_pad_mask = (tgt_batch == PAD_OUT_FOLD)
                    
                    src = src_batch.transpose(0, 1)
                    tgt = tgt_batch.transpose(0, 1)
                    tgt_input = tgt[:-1, :]
                    tgt_output = tgt[1:, :]
                    
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                        tgt_input.size(0)
                    ).to(device)
                    
                    logits = model_fold(
                        src,
                        tgt_input,
                        tgt_mask=tgt_mask,
                        src_key_padding_mask=src_pad_mask,
                        tgt_key_padding_mask=tgt_pad_mask[:, :-1]
                    )
                    
                    loss = criterion_fold(
                        logits.reshape(-1, logits.size(-1)),
                        tgt_output.reshape(-1)
                    )
                    val_loss += loss.item()
                
                for i in range(len(val_dataset_fold)):
                    word = val_df_fold.iloc[i]['Word']
                    gold_seg = val_df_fold.iloc[i]['segmentation']
                    
                    seg_str = segment_word_fold(
                        word, model_fold, input2idx_fold, output2idx_fold, 
                        idx2output_fold, max_len=50
                    )
                    
                    if seg_str == gold_seg:
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
    
    run_kfold = '''# Run k-fold cross-validation
kfold_results = run_kfold_cross_validation(
    df=gold_df,
    n_folds=5,
    d_model=D_MODEL,
    ff=FF,
    heads=HEADS,
    layers=LAYERS,
    drop=DROP,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    random_state=42,
    device=device
)

print(f"\\navg exact match rate: {kfold_results['mean_metrics']['exact_match']:.3f} +/- {kfold_results['std_metrics']['exact_match']:.3f}")
print(f"avg validation loss: {kfold_results['mean_metrics']['val_loss']:.4f} +/- {kfold_results['std_metrics']['val_loss']:.4f}")'''
    nb.cells.append(new_code_cell(run_kfold))


def add_evaluation_section(nb):
    """Add evaluation functions and test set evaluation."""
    
    eval_helpers = '''def parse_segmentation(seg_str):
    """Parse segmented string into list of morphemes."""
    if not seg_str:
        return []
    return seg_str.split('+')

def is_correct_prediction(predicted, gold_variants):
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

# Build reverse vocabulary mapping
idx2output = {idx: ch for ch, idx in output2idx.items()}

# Evaluate on test set
records = []
all_words = test_df["Word"].tolist()

print("predicting segmentations for test words...")
for word in all_words:
    seg_str = segment_word(word, model, input2idx, output2idx, idx2output)
    predicted_segments = parse_segmentation(seg_str)
    
    gold_variants = test_df[test_df["Word"] == word]["Gold"].iloc[0] if len(test_df[test_df["Word"] == word]) > 0 else []
    
    correct_exact = is_correct_prediction(predicted_segments, gold_variants)
    split_metrics = split_count_metrics(predicted_segments, gold_variants)
    
    records.append({
        "Word": word,
        "Prediction": predicted_segments,
        "Gold": gold_variants,
        "CorrectExactSeg": correct_exact,
        "CorrectSplitCount": split_metrics["Exact"],
        "SplitCount+1": split_metrics["+1"],
        "SplitCount-1": split_metrics["-1"],
        "SplitCount±1": split_metrics["±1"],
        "OverlapExactAndSplit": correct_exact and split_metrics["Exact"]
    })

results_df = pd.DataFrame(records)

# Compute aggregate metrics
accuracy = results_df["CorrectExactSeg"].mean()
split_exact_acc = results_df["CorrectSplitCount"].mean()
split_plus1_acc = results_df["SplitCount+1"].mean()
split_minus1_acc = results_df["SplitCount-1"].mean()
split_pm1_acc = results_df["SplitCount±1"].mean()
overlap_accuracy = results_df["OverlapExactAndSplit"].mean()

print(f"\\n=== evaluation results ===")
print(f"exact segmentation accuracy: {accuracy:.4f}")
print(f"\\n=== split-count metrics ===")
print(f"split-count (exact):          {split_exact_acc:.4f}")
print(f"split-count (+1):             {split_plus1_acc:.4f}")
print(f"split-count (−1):              {split_minus1_acc:.4f}")
print(f"split-count (±1):              {split_pm1_acc:.4f}")
print(f"overlap (exact ∩ split):      {overlap_accuracy:.4f}")

# Save results
results_output_path = os.path.join(DATA_FOLDER, "transformer_eval_results.csv")
results_df.to_csv(results_output_path, index=False)
print(f"\\nevaluation results saved to {results_output_path}")'''
    nb.cells.append(new_code_cell(test_eval))


def add_example_section(nb):
    """Add example usage."""
    
    example_code = '''# Example segmentations
test_words = ["pikunas", "rikuchkani", "ñichkanchus"]
print("example segmentations:")
for word in test_words:
    segmented = segment_word(word, model, input2idx, output2idx, idx2output)
    print(f"  {word} -> {segmented}")'''
    nb.cells.append(new_code_cell(example_code))


def main():
    """Build and save the refactored notebook."""
    nb = create_notebook()
    
    # Add sections in order
    add_header_section(nb)
    add_imports_section(nb)
    add_config_section(nb)
    add_data_loading_section(nb)
    add_data_analysis_section(nb)
    add_vocabulary_section(nb)
    add_dataset_section(nb)
    add_model_section(nb)
    add_checkpointing_section(nb)
    add_training_section(nb)
    add_inference_section(nb)
    add_kfold_section(nb)
    add_run_kfold_section(nb)
    add_evaluation_section(nb)
    add_example_section(nb)
    
    # Save notebook
    output_path = "transmorpher_refactored.ipynb"
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print(f"Refactored notebook saved to: {output_path}")


if __name__ == "__main__":
    main()

