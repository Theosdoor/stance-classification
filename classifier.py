"""
Stance Classifier using DeBERTa-v3 with LoRA Fine-tuning

4-way classification: Support, Deny, Query, Comment
Input: source_text + reply_text concatenation
Uses W&B for hyperparameter sweeps and diagnostics
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm.auto import tqdm
from dotenv import load_dotenv
import wandb

from data_loader import load_dataset, format_input_with_context, LABEL_TO_ID, ID_TO_LABEL

# Load environment variables
load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

CLASSIFIER_NAME = "stance-classifier"

# Model options
MODEL_OPTIONS = {
    "deberta": "microsoft/deberta-v3-base",
    # "bertweet": "vinai/bertweet-base",
    "bertweet-large": "vinai/bertweet-large", # has 512 max tokens rather than 128
}

# Model-specific max sequence lengths
# Note: bertweet-large supports 512 but we use 256 to balance context vs GPU memory
MAX_LENGTH_OPTIONS = {
    "deberta": 256,
    "bertweet": 128,
    "bertweet-large": 256,
}

# LoRA target module presets per model
# - "minimal": query + value projections only (fastest training)
# - "attention": all attention projections (balanced)
# - "all_linear": all linear layers including FFN (most expressive but slower)
LORA_TARGET_PRESETS = {
    "deberta": {
        "minimal": ["query_proj", "value_proj"],
        "attention": ["query_proj", "key_proj", "value_proj"],
        "all_linear": ["query_proj", "key_proj", "value_proj", "dense"],
    },
    "bertweet": {
        "minimal": ["query", "value"],
        "attention": ["query", "key", "value"],
        "all_linear": ["query", "key", "value", "dense"],
    },
    "bertweet-large": {
        "minimal": ["query", "value"],
        "attention": ["query", "key", "value"],
        "all_linear": ["query", "key", "value", "dense"],
    },
}

NUM_LABELS = 4

# LoRA defaults
LORA_R = 16 # rank
LORA_ALPHA = 32 # alpha scaling
LORA_DROPOUT = 0.1

# Training defaults (can be overridden by W&B sweep)
DEFAULT_CONFIG = {
    "model_name": "bertweet-large",
    "lora_targets": "all_linear",     # "minimal", "attention", or "all_linear"
    "learning_rate": 2e-5,
    "batch_size": 16,
    "num_epochs": 20,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "use_context": True,             # Include context chain in input
    "use_features": True,            # Include features in input
    "loss_type": "weighted_ce",      # "weighted_ce" or "focal"
    "focal_gamma": 2.0,              # Focal loss gamma (only used if loss_type="focal")
    "early_stopping_patience": 3,    # Stop if no improvement for N epochs
}

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Random seed for reproducibility
SEED = 42

# Paths
CHECKPOINT_DIR = "./results/classifier/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================================
# Dataset
# ============================================================================

class StanceDataset(Dataset):
    """
    Dataset for stance classification with thread context and features.
    
    Uses format_input_with_context() from data_loader to create input strings
    with delimiter-marked source, context, parent, and target tweets.
    """
    
    def __init__(self, df, tokenizer, max_length=256, use_context=True, use_features=True):
        self.df = df.reset_index(drop=True)
        self.full_df = df  # Keep reference for context lookup
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_context = use_context
        self.use_features = use_features
        
        # Pre-compute formatted inputs for efficiency
        self.inputs = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            formatted = format_input_with_context(
                row, self.full_df, 
                use_features=use_features, 
                use_context=use_context,
                max_tokens=max_length,
                tokenizer=tokenizer
            )
            self.inputs.append(formatted)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = self.inputs[idx]
        
        # Tokenize the formatted input
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(row['label'], dtype=torch.long)
        }


# ============================================================================
# Model Setup
# ============================================================================

def compute_class_weights(train_df):
    """
    Compute class weights for imbalanced dataset.
    
    Class Imbalance Mitigation:
    - RumourEval has ~70% Comment class (majority)
    - Using inverse frequency weighting: weight[c] = total / (num_classes * count[c])
    - This gives more importance to minority classes (Support, Deny, Query)
    """
    label_counts = train_df['label'].value_counts().sort_index()
    total_samples = len(train_df)
    num_classes = len(label_counts)
    
    weights = []
    for label_id in range(num_classes):
        count = label_counts.get(label_id, 1)
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    
    Focal loss down-weights easy examples and focuses on hard ones.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        weight: Class weights for imbalance (like CE weight)
        gamma: Focusing parameter. Higher = more focus on hard examples.
               gamma=0 is equivalent to weighted cross-entropy.
               Typical values: 1.0, 2.0, 5.0
    """
    
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.weight, reduction='none'
        )
        pt = torch.exp(-ce_loss)  # p_t = probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def create_model(config):
    """Create model with LoRA adapter. Supports DeBERTa-v3 and BERTweet."""
    
    # Resolve model name
    model_key = config.get("model_name", "deberta")
    model_path = MODEL_OPTIONS.get(model_key)
    
    # Resolve LoRA target modules
    lora_targets_key = config.get("lora_targets", "attention")
    target_modules = LORA_TARGET_PRESETS[model_key].get(
        lora_targets_key, 
        LORA_TARGET_PRESETS[model_key]["attention"]
    )
    
    print(f"Loading model: {model_path}")
    print(f"LoRA target modules: {target_modules}")
    
    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=NUM_LABELS,
        problem_type="single_label_classification"
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.get("lora_r", LORA_R),
        lora_alpha=config.get("lora_alpha", LORA_ALPHA),
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        bias="none",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model.to(DEVICE), model_path


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, criterion, scaler=None):
    """Train for one epoch with optional mixed precision."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    use_amp = scaler is not None
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast(enabled=use_amp):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            logits = outputs.logits
            loss = criterion(logits, labels)
        
        # Backward with scaler if using AMP
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, macro_f1


def evaluate(model, dataloader, criterion):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            logits = outputs.logits
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, macro_f1, all_preds, all_labels


def train(config=None):
    """
    Full training pipeline with W&B logging.
    
    Logs:
    - Training loss per epoch
    - Validation loss per epoch
    - Validation macro-F1 per epoch
    - Best model checkpoint
    """
    
    # Initialize W&B
    with wandb.init(config=config):
        config = wandb.config
        
        print(f"\n{'='*60}")
        print(f"Training with config: {dict(config)}")
        print(f"Device: {DEVICE}")
        print(f"{'='*60}\n")
        
        # Set seed
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        
        # Load data
        print("Loading data...")
        train_df, dev_df, _ = load_dataset()
        
        print(f"Train samples: {len(train_df)}")
        print(f"Dev samples: {len(dev_df)}")
        print(f"\nClass distribution (train):")
        print(train_df['label_text'].value_counts())
        
        # Compute class weights for imbalance
        class_weights = compute_class_weights(train_df).to(DEVICE)
        print(f"\nClass weights: {class_weights.tolist()}")
        
        # Initialize tokenizer
        model_key = config.get("model_name", "deberta")
        model_path = MODEL_OPTIONS.get(model_key)
        max_length = MAX_LENGTH_OPTIONS.get(model_key, 256)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        
        # Get context/feature flags from config
        use_context = config.get("use_context", True)
        use_features = config.get("use_features", True)
        print(f"\nUsing context: {use_context}, Using features: {use_features}")
        
        # Create datasets with model-appropriate max length and context/features
        train_dataset = StanceDataset(
            train_df, tokenizer, max_length=max_length,
            use_context=use_context, use_features=use_features
        )
        dev_dataset = StanceDataset(
            dev_df, tokenizer, max_length=max_length,
            use_context=use_context, use_features=use_features
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            num_workers=0
        )
        dev_loader = DataLoader(
            dev_dataset, 
            batch_size=config.batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        # Create model
        model, _ = create_model(config)
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        total_steps = len(train_loader) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss function - weighted CE or focal loss for class imbalance
        loss_type = config.get("loss_type", "weighted_ce")
        if loss_type == "focal":
            focal_gamma = config.get("focal_gamma", 2.0)
            criterion = FocalLoss(weight=class_weights, gamma=focal_gamma)
            print(f"Using Focal Loss (gamma={focal_gamma})")
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print("Using Weighted Cross-Entropy Loss")
        
        # Mixed precision scaler for faster training (CUDA only)
        use_amp = DEVICE.type == "cuda"
        scaler = GradScaler() if use_amp else None
        if use_amp:
            print("Using mixed precision training (fp16)")
        
        # Early stopping setup
        patience = config.get("early_stopping_patience", 3)
        epochs_without_improvement = 0
        
        # Training loop
        best_f1 = 0
        best_epoch = 0
        
        for epoch in range(config.num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{config.num_epochs} ---")
            
            # Train
            train_loss, train_f1 = train_epoch(
                model, train_loader, optimizer, scheduler, criterion, scaler
            )
            
            # Evaluate
            val_loss, val_f1, val_preds, val_labels = evaluate(
                model, dev_loader, criterion
            )
            
            # Log to W&B
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/macro_f1": train_f1,
                "val/loss": val_loss,
                "val/macro_f1": val_f1,
                "learning_rate": scheduler.get_last_lr()[0]
            })
            
            print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                
                # Save checkpoint
                checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model")
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                
                print(f"New best model saved! F1: {best_f1:.4f}")
                
                # Log confusion matrix for best model
                cm = confusion_matrix(val_labels, val_preds)
                wandb.log({
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=val_labels,
                        preds=val_preds,
                        class_names=list(LABEL_TO_ID.keys())
                    )
                })
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} epoch(s)")
                
                if epochs_without_improvement >= patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
        
        # Final report
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Validation F1: {best_f1:.4f} (Epoch {best_epoch})")
        print(f"{'='*60}")
        
        # Log final metrics
        wandb.log({
            "best_val_f1": best_f1,
            "best_epoch": best_epoch
        })
        
        # Print classification report for best epoch
        print("\nClassification Report (last epoch):")
        print(classification_report(
            val_labels, val_preds, 
            target_names=list(LABEL_TO_ID.keys())
        ))
        
        return best_f1


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train stance classifier")
    parser.add_argument("--agent", action="store_true", help="Run as W&B agent worker (for wandb agent <sweep_id>)")
    args = parser.parse_args()
    
    if args.agent:
        # Called by: wandb agent <sweep_id>
        # W&B agent handles wandb.init() and passes config
        print("Running as W&B agent worker...")
        train()
    else:
        print("Starting single training run with default config...")
        wandb.init(
            project="NLP_cswk",
            config=DEFAULT_CONFIG,
        )
        train(DEFAULT_CONFIG)
        wandb.finish()