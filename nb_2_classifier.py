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
WARMUP_RATIO = 0.15
WEIGHT_DECAY = 0.05
EARLY_STOP_VAL_F1 = 0.68 # only stop early if the val macro-f1 is above this value
EARLY_STOPPING_PATIENCE = 3

MAX_LENGTH = 256
N_LABELS = len(LABEL2ID) # 4 SDQC

LORA_TARGET_MODULES = ["query", "value"]
LORA_ALPHA = 48
LORA_R = 20
LORA_DROPOUT = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
RAND_SEED = 42 # for reproducing results
torch.manual_seed(RAND_SEED)
np.random.seed(RAND_SEED)

CHECKPOINT_DIR = "./results/classifier/special"
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
# helper functions for CSV I/O (reused from prompting_experiments.py)

def load_or_create_csv(path, columns):
    """Load existing CSV or create empty DataFrame with columns."""
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns)

def append_row_to_csv(path, row_dict):
    """Append a single row to CSV file."""
    df = pd.DataFrame([row_dict])
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)


# %%
# plotting functions (work from saved CSVs)

def plot_training_diagnostics(csv_path=None, save_path=None):
    """Plot training diagnostics from saved CSV."""
    if csv_path is None:
        csv_path = os.path.join(CHECKPOINT_DIR, "training_logs.csv")
    if save_path is None:
        save_path = os.path.join(CHECKPOINT_DIR, "train_diag.png")
    
    logs_df = pd.read_csv(csv_path)
    epochs = range(1, len(logs_df) + 1)
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 5))
    
    # loss
    plt.subplot(1, 2, 1)
    sns.lineplot(x=list(epochs), y=logs_df['train_loss'], marker='o', label='Train Loss')
    sns.lineplot(x=list(epochs), y=logs_df['val_loss'], marker='o', color='orange', label='Val Loss')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim(0)
    
    # f1
    plt.subplot(1, 2, 2)
    sns.lineplot(x=list(epochs), y=logs_df['train_f1'], marker='o', label='Train Macro F1')
    sns.lineplot(x=list(epochs), y=logs_df['val_f1'], marker='o', color='orange', label='Val Macro F1')
    plt.title('Macro F1 vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Macro F1 Score')
    plt.xlim(0)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training diagnostics plot saved to {save_path}")
    plt.close()


def plot_test_metrics_heatmap(csv_path=None, save_path=None):
    """Plot test metrics heatmap from saved CSV."""
    if csv_path is None:
        csv_path = os.path.join(CHECKPOINT_DIR, "test_metrics.csv")
    if save_path is None:
        save_path = os.path.join(CHECKPOINT_DIR, "test_metrics_heatmap.png")
    
    df_plot = pd.read_csv(csv_path, index_col=0)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_plot, annot=True, fmt='.3f')
    plt.title('Test Metrics: Per-class & Macro F1')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Test metrics heatmap saved to {save_path}")
    plt.close()


def plot_confusion_matrix_from_csv(csv_path=None, save_path=None):
    """Plot confusion matrix from saved CSV."""
    if csv_path is None:
        csv_path = os.path.join(CHECKPOINT_DIR, "confusion_matrix.csv")
    if save_path is None:
        save_path = os.path.join(CHECKPOINT_DIR, "test_confusion_matrix.png")
    
    cm = pd.read_csv(csv_path, index_col=0).values
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(LABEL2ID.keys()), 
                yticklabels=list(LABEL2ID.keys()))
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix heatmap saved to {save_path}")
    plt.close()


def generate_all_classifier_plots():
    """Generate all plots from saved CSVs (can run separately after experiments)."""
    print("Generating classifier plots from saved CSVs...")
    
    # Training diagnostics
    train_csv = os.path.join(CHECKPOINT_DIR, "training_logs.csv")
    if os.path.exists(train_csv):
        plot_training_diagnostics(train_csv)
    
    # Test metrics heatmap
    metrics_csv = os.path.join(CHECKPOINT_DIR, "test_metrics.csv")
    if os.path.exists(metrics_csv):
        plot_test_metrics_heatmap(metrics_csv)
    
    # Confusion matrix
    cm_csv = os.path.join(CHECKPOINT_DIR, "confusion_matrix.csv")
    if os.path.exists(cm_csv):
        plot_confusion_matrix_from_csv(cm_csv)
    
    print("All classifier plots saved.")


# %%
# pipeline

def train(train_df, dev_df, verbose=True):        
    # get class weights for loss function
    class_weights = compute_class_weights(train_df).to(DEVICE)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
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
    
    logs = {'train_loss': [], 'train_f1': [], 'val_loss': [], 'val_f1': []}
    
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
        logs['train_f1'].append(train_f1)
        logs['val_loss'].append(val_loss)
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
        model, tokenizer, logs = train(train_df, dev_df)
        
        # save training logs to CSV
        logs_csv_path = os.path.join(CHECKPOINT_DIR, "training_logs.csv")
        logs_df = pd.DataFrame(logs)
        logs_df.to_csv(logs_csv_path, index=False)
        print(f"Training logs saved to {logs_csv_path}")
        
        # plot from saved CSV
        plot_training_diagnostics(logs_csv_path)
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
    
    # report
    report_str = classification_report(
        test_labels, test_preds,
        target_names=list(LABEL2ID.keys())
    )
    print(report_str)
    
    # save classification report to TXT
    report_txt_path = os.path.join(CHECKPOINT_DIR, "classifier_classification_report.txt")
    with open(report_txt_path, 'w') as f:
        f.write("Classification Report: classifier\n")
        f.write("=" * 50 + "\n")
        f.write(report_str)
    print(f"Classification report saved to {report_txt_path}")
    
    # save classification report to CSV
    report_dict_full = classification_report(
        test_labels, test_preds,
        target_names=list(LABEL2ID.keys()),
        output_dict=True
    )
    report_df_full = pd.DataFrame(report_dict_full).T
    report_csv_path = os.path.join(CHECKPOINT_DIR, "classifier_classification_report.csv")
    report_df_full.to_csv(report_csv_path)
    print(f"Classification report CSV saved to {report_csv_path}")
    
    # save test metrics to CSV
    report_dict = classification_report(
        test_labels, test_preds, 
        target_names=list(LABEL2ID.keys()), 
        output_dict=True
    )
    
    df_report = pd.DataFrame(report_dict).transpose()
    rows_to_keep = list(LABEL2ID.keys()) + ['macro avg']
    cols_to_keep = ['precision', 'recall', 'f1-score']
    df_plot = df_report.loc[rows_to_keep, cols_to_keep]
    
    metrics_csv_path = os.path.join(CHECKPOINT_DIR, "test_metrics.csv")
    df_plot.to_csv(metrics_csv_path)
    print(f"Test metrics saved to {metrics_csv_path}")
    
    # plot from saved CSV
    plot_test_metrics_heatmap(metrics_csv_path)

    # save confusion matrix to CSV
    cm = confusion_matrix(test_labels, test_preds)
    print(cm)
    
    cm_df = pd.DataFrame(cm, index=list(LABEL2ID.keys()), columns=list(LABEL2ID.keys()))
    cm_csv_path = os.path.join(CHECKPOINT_DIR, "confusion_matrix.csv")
    cm_df.to_csv(cm_csv_path)
    print(f"Confusion matrix saved to {cm_csv_path}")
    
    # plot from saved CSV
    plot_confusion_matrix_from_csv(cm_csv_path)
    
    # generate exp4-compatible classifier results CSV
    labels = ['support', 'deny', 'query', 'comment']
    per_class_f1 = f1_score(test_labels, test_preds, average=None, labels=[LABEL2ID[l] for l in labels])
    
    classifier_row = {
        'strategy': 'classifier',
        'repeat': 1,
        'macro_f1': test_f1,
        'support_f1': per_class_f1[0],
        'deny_f1': per_class_f1[1],
        'query_f1': per_class_f1[2],
        'comment_f1': per_class_f1[3],
    }
    
    exp4_csv_path = os.path.join(CHECKPOINT_DIR, "classifier_exp4_results.csv")
    pd.DataFrame([classifier_row]).to_csv(exp4_csv_path, index=False)
    print(f"Exp4-compatible classifier results saved to {exp4_csv_path}")