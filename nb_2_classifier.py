# %%
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_dataset, format_input_with_context, LABEL2ID, ID2LABEL

# %% [markdown]
# # Training

# %%
# hyperparameters

MODEL_NAME = "vinai/bertweet-large"
BATCH_SIZE = 8
LEARNING_RATE = 0.00004
NUM_EPOCHS = 20
WARMUP_RATIO = 0.2
WEIGHT_DECAY = 0.03
EARLY_STOP_VAL_F1 = 0.7 # only stop early if the val macro-f1 is above this value
EARLY_STOPPING_PATIENCE = 3

MAX_LENGTH = 512
N_LABELS = len(LABEL2ID) # 4 SDQC

LORA_TARGET_MODULES = ["query", "key", "value"]
LORA_ALPHA = 64
LORA_R = 16
LORA_DROPOUT = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
RAND_SEED = 42 # for reproducing results
torch.manual_seed(RAND_SEED)
np.random.seed(RAND_SEED)

CHECKPOINT_DIR = "./results/classifier/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# %%
# dataset

class InputDataset(Dataset):
    """
    preload with format_input_with_context() to create classifier inputs    
    """ 
    def __init__(self, df, tokenizer, max_length=256, use_context=True, use_features=True):
        self.df = df.reset_index(drop=True)
        self.full_df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_context = use_context
        self.use_features = use_features
        
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

# %%
# model

def compute_class_weights(train_df):
    """
    use inverse frequency weighting to give minority classes more importance
        weight for class i = total_samples / (num_classes * count_i)
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

def create_model():    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=N_LABELS,
        problem_type="single_label_classification"
    )
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model.to(DEVICE)

# %%
# train helpers

def train_epoch(model, dataloader, optimizer, scheduler, criterion, scaler=None):
    """train for one epoch with optional mixed precision"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    use_amp = scaler is not None
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        optimizer.zero_grad()
        
        # use autocast for mixed precision in forward pass
        with autocast(device_type=DEVICE.type, enabled=use_amp):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            logits = outputs.logits
            loss = criterion(logits, labels)
        
        # use scaler for mixed precision in backward pass
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


def evaluate(model, dataloader, loss_fn):
    """Evaluate model on dev / test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, macro_f1, all_preds, all_labels

# %%
# pipeline

def train(train_df, dev_df, verbose=True):    
    print(f"\n{'='*60}")
    print(f"Training with:")
    print(f"  BATCH_SIZE={BATCH_SIZE}, LEARNING_RATE={LEARNING_RATE}, NUM_EPOCHS={NUM_EPOCHS}")
    print(f"  LORA_R={LORA_R}, LORA_ALPHA={LORA_ALPHA}")
    print(f"  WARMUP_RATIO={WARMUP_RATIO}, WEIGHT_DECAY={WEIGHT_DECAY}")
    print(f"  EARLY_STOPPING_PATIENCE={EARLY_STOPPING_PATIENCE}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}\n")
    
    # get class weights for loss function
    class_weights = compute_class_weights(train_df).to(DEVICE)
    
    # tokenizer and load datasets
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    
    train_dataset = InputDataset(
        train_df, tokenizer, max_length=MAX_LENGTH,
        use_context=True, use_features=True
    )
    dev_dataset = InputDataset(
        dev_df, tokenizer, max_length=MAX_LENGTH,
        use_context=True, use_features=True
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
    )
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
    )
    
    # init model
    model = create_model()
    
    # init optimizer, scheduler, loss fn
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # use weighted cross-entropy loss
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    # mixed precision on cuda
    use_amp = DEVICE.type == "cuda"
    scaler = GradScaler(device=DEVICE.type) if use_amp else None
    
    # train loop
    best_f1 = 0
    epochs_without_improvement = 0
    
    logs = {'train_loss': [], 'val_f1': []}
    
    for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):        
        # train
        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, scaler
        )
        
        # eval
        val_loss, val_f1, _, _ = evaluate(
            model, dev_loader, loss_fn
        )
        
        logs['train_loss'].append(train_loss)
        logs['val_f1'].append(val_f1)

        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
        
        # save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_without_improvement = 0
            
            # save chekckpoint
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model")
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            
            print(f"New model saved! F1: {best_f1:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs")
            
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE and val_f1 >= EARLY_STOP_VAL_F1:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs (val F1 >= {EARLY_STOP_VAL_F1})")
                break
    
    return model, tokenizer, logs

# %% [markdown]
# # Analysis

# %%
# load / train and evaluate
if __name__ == "__main__":
    # load data
    train_df, dev_df, test_df = load_dataset()

    # train if no checkpoint exists, else load from checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model")
    if not os.path.exists(checkpoint_path):
        print("No checkpoint found, training new model...")
        model, tokenizer, history = train(train_df, dev_df)
        
        # plot training diagnostics
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(12, 5))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        plt.subplot(1, 2, 1)
        sns.lineplot(x=epochs, y=history['train_loss'], marker='o', label='Train Loss')
        plt.title('Training Loss vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        sns.lineplot(x=epochs, y=history['val_f1'], marker='o', color='orange', label='Val Macro F1')
        plt.title('Validation Macro F1 vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Macro F1')
        
        plt.tight_layout()
        plot_path = os.path.join(CHECKPOINT_DIR, "train_diag.png")
        plt.savefig(plot_path)
        print(f"Training diagnostics plot saved to {plot_path}")
        plt.show()
    else:
        print(f"Loading existing checkpoint at {checkpoint_path}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=False)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=N_LABELS,
            problem_type="single_label_classification"
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model = model.to(DEVICE)
        
    # evaluate on test set
    test_dataset = InputDataset(
        test_df, tokenizer, max_length=MAX_LENGTH,
        use_context=True, use_features=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    class_weights = compute_class_weights(train_df).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    test_loss, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, loss_fn
    )

    print(f"Test Loss: {test_loss:.4f} | Test F1: {test_f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(
        test_labels, test_preds,
        target_names=list(LABEL2ID.keys())
    ))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))