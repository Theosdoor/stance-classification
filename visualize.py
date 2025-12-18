import json
import matplotlib.pyplot as plt
import os
import argparse
import pandas as pd

def plot_metrics(log_file, output_dir):
    with open(log_file, 'r') as f:
        data = json.load(f)
        
    history = data['log_history']
    df = pd.DataFrame(history)
    
    # Filter for training loss
    train_loss = df[df['loss'].notna()][['step', 'loss', 'epoch']]
    
    # Filter for eval metrics
    eval_metrics = df[df['eval_f1'].notna()][['step', 'eval_f1', 'eval_loss', 'eval_accuracy', 'epoch']]
    
    # Plot Training Loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss['epoch'], train_loss['loss'], label='Training Loss', marker='o')
    if not eval_metrics.empty:
        plt.plot(eval_metrics['epoch'], eval_metrics['eval_loss'], label='Validation Loss', marker='x')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()
    
    # Plot F1 Score
    if not eval_metrics.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(eval_metrics['epoch'], eval_metrics['eval_f1'], label='Macro F1', color='green', marker='s')
        plt.title('Validation Macro F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'f1_curve.png'))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, default='results/checkpoint-744/trainer_state.json')
    parser.add_argument('--output_dir', type=str, default='results/plots')
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    plot_metrics(args.log_file, args.output_dir)
