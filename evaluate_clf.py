"""
Classifier Evaluation Script

This script:
1. Fetches W&B run statistics for the NLP_cswk project
2. Loads the saved model checkpoint to show example predictions
3. Provides analysis and suggestions for improvement
"""

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

from data_loader import get_dev_data, get_test_data, LABEL_TO_ID, ID_TO_LABEL

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

CHECKPOINT_DIR = "./results/classifier/best_model"


# ============================================================================
# W&B Statistics
# ============================================================================

def fetch_wandb_statistics():
    """Fetch and analyze W&B run statistics."""
    print("=" * 70)
    print("WEIGHTS & BIASES RUN STATISTICS")
    print("=" * 70)
    
    api = wandb.Api()
    runs = api.runs("theo-farrell99-durham-university/NLP_cswk")
    
    results = []
    for run in runs:
        summary = run.summary._json_dict
        config = {k: v for k, v in run.config.items() if not k.startswith('_')}
        
        # Extract key metrics
        result = {
            'name': run.name,
            'state': run.state,
            'best_val_f1': summary.get('best_val_f1', None),
            'best_epoch': summary.get('best_epoch', None),
            'final_train_loss': summary.get('train/loss', None),
            'final_val_loss': summary.get('val/loss', None),
            'final_val_f1': summary.get('val/macro_f1', None),
            'runtime_seconds': summary.get('_runtime', None),
            'model_name': config.get('model_name', None),
            'lora_targets': config.get('lora_targets', None),
            'learning_rate': config.get('learning_rate', None),
            'batch_size': config.get('batch_size', None),
            'num_epochs': config.get('num_epochs', None),
            'lora_r': config.get('lora_r', None),
            'lora_alpha': config.get('lora_alpha', None),
        }
        results.append(result)
    
    df = pd.DataFrame(results)
    
    # Filter to completed runs with metrics
    completed = df[df['best_val_f1'].notna()].copy()
    
    if len(completed) == 0:
        print("No completed runs with metrics found!")
        return df
    
    # Sort by best_val_f1
    completed = completed.sort_values('best_val_f1', ascending=False)
    
    print(f"\n📊 Total runs: {len(df)}")
    print(f"✅ Completed runs with metrics: {len(completed)}")
    
    print("\n" + "-" * 70)
    print("TOP 10 RUNS BY VALIDATION F1")
    print("-" * 70)
    
    top_cols = ['name', 'best_val_f1', 'best_epoch', 'model_name', 'lora_targets', 
                'learning_rate', 'batch_size', 'lora_r', 'lora_alpha']
    print(completed[top_cols].head(10).to_string(index=False))
    
    # Best run analysis
    best_run = completed.iloc[0]
    print("\n" + "-" * 70)
    print("🏆 BEST RUN ANALYSIS")
    print("-" * 70)
    print(f"Run Name: {best_run['name']}")
    print(f"Best Val F1: {best_run['best_val_f1']:.4f}")
    print(f"Best Epoch: {best_run['best_epoch']}")
    print(f"Final Train Loss: {best_run['final_train_loss']:.4f}" if best_run['final_train_loss'] else "N/A")
    print(f"Final Val Loss: {best_run['final_val_loss']:.4f}" if best_run['final_val_loss'] else "N/A")
    print(f"Runtime: {best_run['runtime_seconds']/60:.1f} minutes" if best_run['runtime_seconds'] else "N/A")
    print("\nConfiguration:")
    print(f"  Model: {best_run['model_name']}")
    print(f"  LoRA Targets: {best_run['lora_targets']}")
    print(f"  Learning Rate: {best_run['learning_rate']}")
    print(f"  Batch Size: {best_run['batch_size']}")
    print(f"  LoRA r: {best_run['lora_r']}")
    print(f"  LoRA alpha: {best_run['lora_alpha']}")
    
    # Statistical summary
    print("\n" + "-" * 70)
    print("PERFORMANCE STATISTICS ACROSS COMPLETED RUNS")
    print("-" * 70)
    print(f"Val F1 Mean: {completed['best_val_f1'].mean():.4f}")
    print(f"Val F1 Std:  {completed['best_val_f1'].std():.4f}")
    print(f"Val F1 Min:  {completed['best_val_f1'].min():.4f}")
    print(f"Val F1 Max:  {completed['best_val_f1'].max():.4f}")
    
    # Hyperparameter impact analysis
    print("\n" + "-" * 70)
    print("HYPERPARAMETER IMPACT ANALYSIS")
    print("-" * 70)
    
    # Model comparison
    if 'model_name' in completed.columns and completed['model_name'].nunique() > 1:
        model_perf = completed.groupby('model_name')['best_val_f1'].agg(['mean', 'std', 'count'])
        print("\nBy Model:")
        print(model_perf.to_string())
    
    # LoRA targets comparison
    if 'lora_targets' in completed.columns and completed['lora_targets'].nunique() > 1:
        lora_perf = completed.groupby('lora_targets')['best_val_f1'].agg(['mean', 'std', 'count'])
        print("\nBy LoRA Targets:")
        print(lora_perf.to_string())
    
    # Learning rate correlation
    if completed['learning_rate'].nunique() > 1:
        lr_corr = completed[['learning_rate', 'best_val_f1']].corr().iloc[0, 1]
        print(f"\nLearning Rate vs Val F1 Correlation: {lr_corr:.3f}")
    
    return df


# ============================================================================
# Model Loading & Inference
# ============================================================================

def load_model():
    """Load the saved model checkpoint."""
    print("\n" + "=" * 70)
    print("LOADING SAVED MODEL CHECKPOINT")
    print("=" * 70)
    
    # Read adapter config to get base model
    config_path = os.path.join(CHECKPOINT_DIR, "adapter_config.json")
    with open(config_path, 'r') as f:
        adapter_config = json.load(f)
    
    base_model_name = adapter_config['base_model_name_or_path']
    print(f"Base model: {base_model_name}")
    print(f"LoRA rank: {adapter_config['r']}")
    print(f"LoRA alpha: {adapter_config['lora_alpha']}")
    print(f"Target modules: {adapter_config['target_modules']}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR, use_fast=False)
    
    # Load base model with safetensors preference
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=4,
        problem_type="single_label_classification",
        use_safetensors=True,  # Prefer safetensors format
    )
    
    # Load LoRA adapter with safetensors explicitly
    model = PeftModel.from_pretrained(
        base_model, 
        CHECKPOINT_DIR,
        safe_serialization=True,  # Use safetensors
    )
    model = model.to(DEVICE)
    model.eval()
    
    print(f"✅ Model loaded successfully on {DEVICE}")
    
    return model, tokenizer


def predict(model, tokenizer, source_text, reply_text, max_length=128):
    """Make a single prediction."""
    encoding = tokenizer(
        source_text,
        reply_text,
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1).item()
    
    return {
        'prediction': ID_TO_LABEL[pred],
        'confidence': probs[0, pred].item(),
        'probabilities': {ID_TO_LABEL[i]: probs[0, i].item() for i in range(4)}
    }


def evaluate_on_dataset(model, tokenizer, df, max_length=128, desc="Evaluating"):
    """Evaluate model on a dataset."""
    predictions = []
    labels = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        result = predict(model, tokenizer, row['source_text'], row['reply_text'], max_length)
        predictions.append(LABEL_TO_ID[result['prediction']])
        labels.append(row['label'])
    
    return predictions, labels


def show_example_predictions(model, tokenizer, df, num_examples=10):
    """Show example predictions with their ground truth."""
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTIONS")
    print("=" * 70)
    
    # Sample from each class if possible
    examples = df.groupby('label_text').apply(
        lambda x: x.sample(min(3, len(x)), random_state=42)
    ).reset_index(drop=True)
    
    if len(examples) > num_examples:
        examples = examples.sample(num_examples, random_state=42)
    
    for i, (_, row) in enumerate(examples.iterrows()):
        print(f"\n--- Example {i+1} ---")
        print(f"Source: {row['source_text'][:150]}...")
        print(f"Reply:  {row['reply_text'][:150]}...")
        print(f"Ground Truth: {row['label_text'].upper()}")
        
        result = predict(model, tokenizer, row['source_text'], row['reply_text'])
        
        is_correct = result['prediction'] == row['label_text']
        status = "✅" if is_correct else "❌"
        
        print(f"Prediction:   {result['prediction'].upper()} ({result['confidence']:.1%}) {status}")
        print(f"Probabilities: S={result['probabilities']['support']:.2f} "
              f"D={result['probabilities']['deny']:.2f} "
              f"Q={result['probabilities']['query']:.2f} "
              f"C={result['probabilities']['comment']:.2f}")


def full_evaluation(model, tokenizer):
    """Run full evaluation on dev set."""
    print("\n" + "=" * 70)
    print("FULL EVALUATION ON DEV SET")
    print("=" * 70)
    
    dev_df = get_dev_data()
    print(f"Dev set size: {len(dev_df)}")
    print(f"Class distribution:")
    print(dev_df['label_text'].value_counts())
    
    predictions, labels = evaluate_on_dataset(model, tokenizer, dev_df, desc="Dev set")
    
    # Classification report
    print("\n" + "-" * 70)
    print("CLASSIFICATION REPORT")
    print("-" * 70)
    print(classification_report(labels, predictions, target_names=list(LABEL_TO_ID.keys())))
    
    # Confusion matrix
    print("\n" + "-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)
    cm = confusion_matrix(labels, predictions)
    cm_df = pd.DataFrame(cm, 
                         index=[f"True:{l}" for l in LABEL_TO_ID.keys()],
                         columns=[f"Pred:{l}" for l in LABEL_TO_ID.keys()])
    print(cm_df)
    
    # Per-class analysis
    print("\n" + "-" * 70)
    print("PER-CLASS ERROR ANALYSIS")
    print("-" * 70)
    
    dev_df['predicted'] = [ID_TO_LABEL[p] for p in predictions]
    dev_df['correct'] = dev_df['predicted'] == dev_df['label_text']
    
    for label in LABEL_TO_ID.keys():
        class_df = dev_df[dev_df['label_text'] == label]
        if len(class_df) == 0:
            continue
        
        correct = class_df['correct'].sum()
        total = len(class_df)
        accuracy = correct / total
        
        print(f"\n{label.upper()} (n={total}, acc={accuracy:.1%}):")
        
        # Show misclassification breakdown
        errors = class_df[~class_df['correct']]
        if len(errors) > 0:
            error_dist = errors['predicted'].value_counts()
            print(f"  Misclassified as: {dict(error_dist)}")
    
    macro_f1 = f1_score(labels, predictions, average='macro')
    
    return macro_f1, predictions, labels


# ============================================================================
# Improvement Suggestions
# ============================================================================

def generate_suggestions(wandb_df, macro_f1):
    """Generate suggestions for improving the classifier."""
    print("\n" + "=" * 70)
    print("🎯 SUGGESTIONS FOR IMPROVEMENT")
    print("=" * 70)
    
    suggestions = []
    
    # Based on current performance
    if macro_f1 < 0.45:
        suggestions.append({
            'priority': 'HIGH',
            'area': 'Model Architecture',
            'suggestion': 'Current F1 is below 0.45. Consider trying larger models like DeBERTa-large or ensemble approaches.',
        })
    
    if macro_f1 < 0.50:
        suggestions.append({
            'priority': 'HIGH',
            'area': 'Data Augmentation',
            'suggestion': 'Add data augmentation techniques: back-translation, synonym replacement, or use external rumour datasets.',
        })
    
    # Based on W&B runs analysis
    completed = wandb_df[wandb_df['best_val_f1'].notna()]
    
    if len(completed) > 0:
        # Check if hyperparameter exploration was thorough
        lr_range = completed['learning_rate'].dropna()
        if len(lr_range.unique()) < 5:
            suggestions.append({
                'priority': 'MEDIUM',
                'area': 'Hyperparameter Search',
                'suggestion': 'Explore wider learning rate range, especially lower values (1e-6 to 5e-5) for fine-tuning.',
            })
        
        # Check lora_r values
        lora_r_range = completed['lora_r'].dropna()
        if lora_r_range.max() < 32:
            suggestions.append({
                'priority': 'MEDIUM',
                'area': 'LoRA Configuration',
                'suggestion': 'Try higher LoRA rank (r=32 or r=64) for more expressivity.',
            })
    
    # General suggestions
    suggestions.extend([
        {
            'priority': 'MEDIUM',
            'area': 'Training Strategy',
            'suggestion': 'Implement label smoothing (0.1) to prevent overconfidence.',
        },
        {
            'priority': 'MEDIUM',
            'area': 'Class Imbalance',
            'suggestion': 'Try focal loss instead of weighted cross-entropy for better minority class handling.',
        },
        {
            'priority': 'MEDIUM',
            'area': 'Input Representation',
            'suggestion': 'Add structured markers like [SOURCE] and [REPLY] tokens to help model distinguish input segments.',
        },
        {
            'priority': 'LOW',
            'area': 'Ensemble',
            'suggestion': 'Create an ensemble of DeBERTa and BERTweet predictions for robustness.',
        },
        {
            'priority': 'LOW',
            'area': 'Context',
            'suggestion': 'Incorporate thread context (previous replies) for better stance understanding.',
        },
    ])
    
    for s in suggestions:
        priority_emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}[s['priority']]
        print(f"\n{priority_emoji} [{s['priority']}] {s['area']}")
        print(f"   {s['suggestion']}")
    
    return suggestions


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("🔍 CLASSIFIER EVALUATION REPORT")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print("=" * 70)
    
    # 1. Fetch W&B statistics
    wandb_df = fetch_wandb_statistics()
    
    # 2. Load model and show examples
    try:
        model, tokenizer = load_model()
        
        # Get dev data for examples
        dev_df = get_dev_data()
        
        # Show example predictions
        show_example_predictions(model, tokenizer, dev_df, num_examples=10)
        
        # Full evaluation
        macro_f1, predictions, labels = full_evaluation(model, tokenizer)
        
        print(f"\n📊 Final Macro F1: {macro_f1:.4f}")
        
    except Exception as e:
        print(f"\n⚠️ Could not load model checkpoint: {e}")
        print("Proceeding with W&B analysis only...")
        macro_f1 = 0.40  # Default for suggestions
    
    # 3. Generate improvement suggestions
    generate_suggestions(wandb_df, macro_f1)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
