# %%
"""
W&B Sweep Analysis and Model Evaluation

This notebook:
1. Fetches W&B run statistics for hyperparameter analysis
2. Analyzes sweep results to find optimal configurations
3. Loads saved model for example predictions
4. Provides improvement suggestions
"""

# %%
# Imports and Setup

import os
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm.auto import tqdm
import wandb

from data_loader import load_dataset, format_input_with_context, LABEL2ID, ID2LABEL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")

CHECKPOINT_DIR = "./results/classifier/best_model"

# %%
# Fetch W&B Runs

api = wandb.Api()
runs = api.runs("theo-farrell99-durham-university/NLP_cswk")

results = []
for run in runs:
    summary = run.summary._json_dict
    config = {k: v for k, v in run.config.items() if not k.startswith('_')}
    
    result = {
        'name': run.name,
        'state': run.state,
        'best_val_f1': summary.get('best_val_f1', None),
        'best_epoch': summary.get('best_epoch', None),
        'final_train_loss': summary.get('train/loss', None),
        'final_val_loss': summary.get('val/loss', None),
        'runtime_seconds': summary.get('_runtime', None),
        'model_name': config.get('model_name', None),
        'lora_targets': config.get('lora_targets', None),
        'learning_rate': config.get('learning_rate', None),
        'batch_size': config.get('batch_size', None),
        'num_epochs': config.get('num_epochs', None),
        'lora_r': config.get('lora_r', None),
        'lora_alpha': config.get('lora_alpha', None),
        'loss_type': config.get('loss_type', None),
        'focal_gamma': config.get('focal_gamma', None),
        'warmup_ratio': config.get('warmup_ratio', None),
        'weight_decay': config.get('weight_decay', None),
    }
    results.append(result)

runs_df = pd.DataFrame(results)
completed_df = runs_df[runs_df['best_val_f1'].notna()].sort_values('best_val_f1', ascending=False)

print(f"Total runs: {len(runs_df)}")
print(f"Completed runs: {len(completed_df)}")

# %%
# Top 10 Runs

print("=" * 70)
print("TOP 10 RUNS BY VALIDATION F1")
print("=" * 70)

top_cols = ['name', 'best_val_f1', 'best_epoch', 'lora_targets', 'learning_rate', 
            'lora_r', 'lora_alpha', 'loss_type', 'warmup_ratio', 'weight_decay']
print(completed_df[top_cols].head(10).to_string(index=False))

# %%
# Best Run Analysis

if len(completed_df) > 0:
    best_run = completed_df.iloc[0]
    
    print("\n" + "=" * 70)
    print("🏆 BEST RUN CONFIGURATION")
    print("=" * 70)
    print(f"Run Name: {best_run['name']}")
    print(f"Best Val F1: {best_run['best_val_f1']:.4f}")
    print(f"Best Epoch: {best_run['best_epoch']}")
    if best_run['runtime_seconds']:
        print(f"Runtime: {best_run['runtime_seconds']/60:.1f} minutes")
    print("\nConfiguration:")
    print(f"  Model: {best_run['model_name']}")
    print(f"  LoRA Targets: {best_run['lora_targets']}")
    print(f"  Learning Rate: {best_run['learning_rate']}")
    print(f"  LoRA r: {best_run['lora_r']}")
    print(f"  LoRA alpha: {best_run['lora_alpha']}")
    print(f"  Loss Type: {best_run['loss_type']}")
    print(f"  Warmup Ratio: {best_run['warmup_ratio']}")
    print(f"  Weight Decay: {best_run['weight_decay']}")

# %%
# Hyperparameter Analysis

print("\n" + "=" * 70)
print("HYPERPARAMETER ANALYSIS")
print("=" * 70)

# By loss type
if completed_df['loss_type'].notna().any():
    print("\nBy Loss Type:")
    print(completed_df.groupby('loss_type')['best_val_f1'].agg(['mean', 'max', 'count']).round(4))

# By LoRA targets
if completed_df['lora_targets'].notna().any():
    print("\nBy LoRA Targets:")
    print(completed_df.groupby('lora_targets')['best_val_f1'].agg(['mean', 'max', 'count']).round(4))

# By LoRA r
if completed_df['lora_r'].notna().any():
    print("\nBy LoRA Rank:")
    print(completed_df.groupby('lora_r')['best_val_f1'].agg(['mean', 'max', 'count']).round(4))

# By Learning Rate
if completed_df['learning_rate'].notna().any():
    print("\nBy Learning Rate:")
    print(completed_df.groupby('learning_rate')['best_val_f1'].agg(['mean', 'max', 'count']).round(4))

# %%
# Performance Statistics

print("\n" + "=" * 70)
print("PERFORMANCE STATISTICS")
print("=" * 70)

if len(completed_df) > 0:
    print(f"Val F1 Mean: {completed_df['best_val_f1'].mean():.4f}")
    print(f"Val F1 Std:  {completed_df['best_val_f1'].std():.4f}")
    print(f"Val F1 Min:  {completed_df['best_val_f1'].min():.4f}")
    print(f"Val F1 Max:  {completed_df['best_val_f1'].max():.4f}")

# %%
# Load Model for Predictions

def load_model():
    """Load the saved model checkpoint."""
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"⚠️ No checkpoint found at {CHECKPOINT_DIR}")
        return None, None
    
    config_path = os.path.join(CHECKPOINT_DIR, "adapter_config.json")
    with open(config_path, 'r') as f:
        adapter_config = json.load(f)
    
    base_model_name = adapter_config['base_model_name_or_path']
    print(f"Loading: {base_model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR, use_fast=False)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=4, problem_type="single_label_classification",
        use_safetensors=True,  # Required for torch < 2.6
    )
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR, safe_serialization=True)
    model = model.to(DEVICE)
    model.eval()
    
    print(f"✅ Model loaded on {DEVICE}")
    return model, tokenizer

try:
    model, tokenizer = load_model()
except Exception as e:
    print(f"⚠️ Could not load model: {e}")
    model, tokenizer = None, None

# %%
# Example Predictions

if model is not None and tokenizer is not None:
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTIONS")
    print("=" * 70)
    
    _, dev_df, _ = load_dataset()
    
    # Sample examples from each class
    examples = dev_df.groupby('label_text').apply(
        lambda x: x.sample(min(2, len(x)), random_state=42)
    ).reset_index(drop=True)
    
    for i, (_, row) in enumerate(examples.head(8).iterrows()):
        print(f"\n--- Example {i+1} ---")
        print(f"Text: {row['text'][:120]}...")
        print(f"Ground Truth: {row['label_text'].upper()}")
        
        input_text = format_input_with_context(row, dev_df, use_features=True, use_context=True)
        
        encoding = tokenizer(input_text, truncation=True, max_length=256, 
                           padding='max_length', return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids=encoding['input_ids'].to(DEVICE),
                          attention_mask=encoding['attention_mask'].to(DEVICE))
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(outputs.logits, dim=1).item()
        
        pred_label = ID2LABEL[pred]
        confidence = probs[0, pred].item()
        status = "✅" if pred_label == row['label_text'] else "❌"
        
        print(f"Prediction: {pred_label.upper()} ({confidence:.1%}) {status}")

# %%
# Quick Dev Set Evaluation

if model is not None and tokenizer is not None:
    print("\n" + "=" * 70)
    print("DEV SET EVALUATION")
    print("=" * 70)
    
    _, dev_df, _ = load_dataset()
    
    predictions = []
    labels = []
    
    for _, row in tqdm(dev_df.iterrows(), total=len(dev_df), desc="Evaluating"):
        input_text = format_input_with_context(row, dev_df, use_features=True, use_context=True)
        
        encoding = tokenizer(input_text, truncation=True, max_length=256, 
                           padding='max_length', return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids=encoding['input_ids'].to(DEVICE),
                          attention_mask=encoding['attention_mask'].to(DEVICE))
            pred = torch.argmax(outputs.logits, dim=1).item()
        
        predictions.append(pred)
        labels.append(row['label'])
    
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=list(LABEL2ID.keys())))
    
    macro_f1 = f1_score(labels, predictions, average='macro')
    print(f"📊 Dev Set Macro F1: {macro_f1:.4f}")

# %%
# Summary

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nTo evaluate on test set, run: python analyse_clf.py")

# %%
