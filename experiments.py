"""
Overnight Experiment Suite for Stance Classification
Run all experiments: python experiments.py
Resume from crash: python experiments.py (automatically skips completed rows)
"""

import os
import re
import json
import torch
import numpy as np
import pandas as pd

# Use non-interactive backend for headless servers (before importing pyplot!)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer
import outlines
from typing import Literal

from data_loader import load_dataset, format_input_with_context


# =============================================================================
# CONFIG
# =============================================================================
MODEL_NAME = 'meta-llama/Llama-3.2-3B-Instruct'
USE_CTX = False
MAX_COT_TOKENS = 200
RESULTS_DIR = "./results/prompting/"
RANDOM_SEED = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)

# Output types for outlines
StanceLabel = Literal["SUPPORT", "DENY", "QUERY", "COMMENT"]
COT_REGEX = outlines.types.Regex(r"[\s\S]*Label: (SUPPORT|DENY|QUERY|COMMENT)")


# =============================================================================
# PROMPTS (from main script)
# =============================================================================
PERSONA = """You are an expert in rumour stance analysis on twitter."""
INSTRUCTION = """Your task is to classify the stance of the [TARGET] tweet towards the veracity of the rumour in the [SRC] tweet."""
INPUT_FORMAT = """\
The input will be provided as a single string containing labeled segments:
"[SRC] ... [PARENT] ... [TARGET] ...". (Note: [PARENT] is omitted if [TARGET] replies directly to [SRC])
"""
LABEL_DEFNS = """\
Classification Labels:
- SUPPORT: The reply explicitly supports or provides evidence for the veracity of the source claim
- DENY: The tweet explicitly denies or provides counter-evidence for the veracity of the source claim
- QUERY: The reply asks for additional evidence in relation to the veracity of the source claim
- COMMENT: The reply makes their own opinionated/neutral comment without a clear contribution to assessing the veracity of the source claim
"""

SYS_PROMPT = f"""\
{PERSONA}
{INSTRUCTION}
{INPUT_FORMAT}
{LABEL_DEFNS}
"""

COT_OUTPUT_FORMAT = """\
Your response should follow this EXACT format:
Reasoning: [VERY BRIEFLY explain why the label fits. Use a two stage strategy: Stage 1 classify Stance vs Non-Stance - If Non-stance, skip stage 2 and answer 'COMMENT'. Otherwise: Stage 2 classify Support / Deny / Query]
Label: [EXACTLY one of: SUPPORT, DENY, QUERY, or COMMENT]
"""

COT_SYS_PROMPT = f"""\
{PERSONA}
{INSTRUCTION}
{INPUT_FORMAT}
{LABEL_DEFNS}
{COT_OUTPUT_FORMAT}
"""

USER_PROMPT_TEMPLATE = """\
Text: {thread_context}

Classify the stance of [TARGET] towards the veracity of the rumour in [SRC]:
"""

# Prompt components for ablation
PROMPT_COMPONENTS = {
    'persona': PERSONA,
    'instruction': INSTRUCTION,
    'input_format': INPUT_FORMAT,
    'label_defns': LABEL_DEFNS,
}

ISOLATED_CONFIGS = {
    'minimal': [],
    '+persona': ['persona'],
    '+instruction': ['instruction'],
    '+input_format': ['input_format'],
    '+label_defns': ['label_defns'],
    'full': ['persona', 'instruction', 'input_format', 'label_defns'],
}

CUMULATIVE_CONFIGS = {
    'minimal': [],
    '+persona': ['persona'],
    '+instruction': ['persona', 'instruction'],
    '+input_format': ['persona', 'instruction', 'input_format'],
    '+label_defns (full)': ['persona', 'instruction', 'input_format', 'label_defns'],
}


# =============================================================================
# FEW-SHOT EXAMPLE SETS
# =============================================================================
DIVERSE_FEW_SHOT_IDS = {
    'support': '553486439129038848',
    'deny': '500384255814299648',
    'query': '525003827494531073',
    'comment': '544314008980172800',
}

SAME_SRC_FEW_SHOT_IDS = {
    'support': '529740822238232577',
    'deny': '529740809672077313',
    'query': '529748991849013248',
    'comment': '529741574742507520',
}

RANDOM_STRATIFIED_FEW_SHOT_IDS = {
    'support': '544306719686656000',
    'deny': '524990163446140928',
    'query': '500386447158161408',
    'comment': '553543395717550080',
}

FEW_SHOT_SETS = {
    'diverse': DIVERSE_FEW_SHOT_IDS,
    'same_src': SAME_SRC_FEW_SHOT_IDS,
    'random': RANDOM_STRATIFIED_FEW_SHOT_IDS,
}

# CoT few-shot examples
COT_FEW_SHOT_IDS = {
    'support': '524967134339022848',
    'deny': '544292581950357504',
    'query': '544281192632053761',
    'comment': '552804023389392896',
}

COT_EXAMPLES = {
    'support': "1. Stance: takes position on veracity\n2. \"must be true\" → affirms claim\nLabel: SUPPORT",
    'deny': "1. Stance: challenges veracity\n2. \"not an Isis flag\", \"false rumors\" → rejects claim\nLabel: DENY",
    'query': "1. Stance: engages with veracity\n2. \"Are your reporters 100% sure?\" → requests evidence\nLabel: QUERY",
    'comment': "1. Stance: no veracity assessment\n2. Offers opinion/reaction only\nLabel: COMMENT",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def load_model():
    """Load model and tokenizer."""
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map='auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = outlines.from_transformers(hf_model, tokenizer)
    model.tokenizer.pad_token = tokenizer.eos_token
    model.tokenizer.padding_side = "left"
    model.model.generation_config.pad_token_id = model.tokenizer.pad_token_id
    return model, tokenizer


def build_user_prompt(thread_context):
    return USER_PROMPT_TEMPLATE.format(thread_context=thread_context)


def build_messages(thread_context, examples=None, sys_prompt=None):
    init_prompt = sys_prompt if sys_prompt is not None else SYS_PROMPT
    messages = [{"role": "system", "content": init_prompt}]
    if examples:
        for ex in examples:
            messages.append({"role": "user", "content": build_user_prompt(ex['source'])})
            messages.append({"role": "assistant", "content": ex.get('response', ex.get('label'))})
    messages.append({"role": "user", "content": build_user_prompt(thread_context)})
    return messages


def build_ablation_sys_prompt(component_keys):
    if not component_keys:
        return ""
    return "\n".join(PROMPT_COMPONENTS[k] for k in component_keys)


def parse_cot_label(text):
    match = re.search(r'Label:\s*(SUPPORT|DENY|QUERY|COMMENT)', text, re.IGNORECASE)
    return match.group(1).lower() if match else None


def get_few_shot_examples(df, n_per_class=1, use_set=None, classes=None):
    """Get few-shot examples with optional class filtering."""
    if classes is None:
        classes = ['support', 'deny', 'query', 'comment']
    examples = []
    
    if use_set is not None:
        id_dict = FEW_SHOT_SETS[use_set]
        for label in classes:
            tweet_id = id_dict[label]
            matches = df[df['tweet_id'] == tweet_id]
            if len(matches) == 0:
                continue
            row = matches.iloc[0]
            examples.append({
                'source': format_input_with_context(row, df, use_features=False, use_context=USE_CTX),
                'label': label.upper()
            })
    else:
        for label in classes:
            class_df = df[df['label_text'] == label]
            samples = class_df.sample(n=min(n_per_class, len(class_df)))
            for _, row in samples.iterrows():
                examples.append({
                    'source': format_input_with_context(row, df, use_features=False, use_context=USE_CTX),
                    'label': label.upper()
                })
    return examples


def get_cot_examples(train_df):
    examples = []
    for label, tweet_id in COT_FEW_SHOT_IDS.items():
        matches = train_df[train_df['tweet_id'] == tweet_id]
        if len(matches) == 0:
            continue
        row = matches.iloc[0]
        examples.append({
            'source': format_input_with_context(row, train_df, use_features=False, use_context=USE_CTX),
            'response': COT_EXAMPLES[label]
        })
    return examples


def evaluate_and_get_metrics(model, tokenizer, df, mode, examples=None, sys_prompt_override=None):
    """Run evaluation and return metrics dict."""
    if mode == "cot":
        output_type = COT_REGEX
        max_new_tokens = MAX_COT_TOKENS
    else:
        output_type = StanceLabel
        max_new_tokens = 10
    
    if sys_prompt_override is not None:
        active_sys_prompt = sys_prompt_override
    elif mode == "cot":
        active_sys_prompt = COT_SYS_PROMPT
    else:
        active_sys_prompt = SYS_PROMPT
    
    predictions = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Eval ({mode})"):
        input_text = format_input_with_context(row, df, use_features=False, use_context=USE_CTX)
        messages = build_messages(input_text, examples=examples, sys_prompt=active_sys_prompt)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = model(prompt, output_type, max_new_tokens=max_new_tokens)
        
        if mode == "cot":
            pred = parse_cot_label(response)
            pred = pred if pred else 'comment'
        else:
            pred = response.lower()
        predictions.append(pred)
    
    true_labels = df['label_text'].tolist()
    labels = ['support', 'deny', 'query', 'comment']
    macro_f1 = f1_score(true_labels, predictions, average='macro')
    per_class_f1 = f1_score(true_labels, predictions, average=None, labels=labels)
    
    return {
        'macro_f1': macro_f1,
        'support_f1': per_class_f1[0],
        'deny_f1': per_class_f1[1],
        'query_f1': per_class_f1[2],
        'comment_f1': per_class_f1[3],
    }


def load_or_create_csv(path, columns):
    """Load existing CSV or create empty DataFrame with columns."""
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns)


def append_row_to_csv(path, row_dict):
    """Append a single row to CSV file."""
    df = pd.DataFrame([row_dict])
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)


# =============================================================================
# EXPERIMENT 1: System Prompt Ablation
# =============================================================================
def run_exp1_ablation(model, tokenizer, train_df, dev_df):
    print("\n" + "="*60)
    print("EXPERIMENT 1: System Prompt Ablation")
    print("="*60)
    
    csv_path = f"{RESULTS_DIR}exp1_ablation.csv"
    columns = ['config_type', 'config_name', 'macro_f1']
    
    existing_df = load_or_create_csv(csv_path, columns)
    completed = set(zip(existing_df.get('config_type', []), existing_df.get('config_name', [])))
    
    all_configs = [
        ('isolated', ISOLATED_CONFIGS),
        ('cumulative', CUMULATIVE_CONFIGS),
    ]
    
    for config_type, configs in all_configs:
        for config_name, component_keys in configs.items():
            if (config_type, config_name) in completed:
                print(f"  Skipping {config_type}/{config_name} (already done)")
                continue
            
            sys_prompt = build_ablation_sys_prompt(component_keys)
            metrics = evaluate_and_get_metrics(model, tokenizer, dev_df, mode="zero-shot", 
                                               sys_prompt_override=sys_prompt)
            
            row = {'config_type': config_type, 'config_name': config_name, 'macro_f1': metrics['macro_f1']}
            append_row_to_csv(csv_path, row)
            print(f"  {config_type}/{config_name}: {metrics['macro_f1']:.4f}")
    
    print(f"Results saved to {csv_path}")
    
    # Generate plots
    results_df = pd.read_csv(csv_path)
    for config_type in ['isolated', 'cumulative']:
        subset = results_df[results_df['config_type'] == config_type]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=subset, y='config_name', x='macro_f1', ax=ax)
        ax.set_xlabel('Macro F1')
        ax.set_ylabel('')
        ax.set_title(f'System Prompt Ablation: {config_type.title()}')
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}exp1_ablation_{config_type}.png", dpi=150)
        plt.close()


# =============================================================================
# EXPERIMENT 2: Few-shot Strategy Comparison
# =============================================================================
def run_exp2_fewshot_strategies(model, tokenizer, train_df, dev_df):
    print("\n" + "="*60)
    print("EXPERIMENT 2: Few-shot Strategy Comparison")
    print("="*60)
    
    csv_path = f"{RESULTS_DIR}exp2_fewshot_strategies.csv"
    columns = ['strategy', 'n_per_class', 'macro_f1', 'support_f1', 'deny_f1', 'query_f1', 'comment_f1']
    
    existing_df = load_or_create_csv(csv_path, columns)
    completed = set(zip(existing_df.get('strategy', []), existing_df.get('n_per_class', [])))
    
    strategies = ['diverse', 'same_src', 'random']
    n_values = [1, 2, 3]
    
    for strategy in strategies:
        for n in n_values:
            if (strategy, n) in completed:
                print(f"  Skipping {strategy}/n={n} (already done)")
                continue
            
            examples = get_few_shot_examples(train_df, n_per_class=n, use_set=strategy)
            metrics = evaluate_and_get_metrics(model, tokenizer, dev_df, mode="few-shot", examples=examples)
            
            row = {'strategy': strategy, 'n_per_class': n, **metrics}
            append_row_to_csv(csv_path, row)
            print(f"  {strategy}/n={n}: macro_f1={metrics['macro_f1']:.4f}")
    
    print(f"Results saved to {csv_path}")
    
    # Generate grouped bar chart
    results_df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=results_df, x='n_per_class', y='macro_f1', hue='strategy', ax=ax)
    ax.set_xlabel('Examples per Class')
    ax.set_ylabel('Macro F1')
    ax.set_title('Few-shot Strategy Comparison')
    ax.legend(title='Strategy')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}exp2_fewshot_strategies.png", dpi=150)
    plt.close()


# =============================================================================
# EXPERIMENT 3: Class Importance Heatmap
# =============================================================================
def run_exp3_class_importance(model, tokenizer, train_df, dev_df):
    print("\n" + "="*60)
    print("EXPERIMENT 3: Class Importance Heatmap")
    print("="*60)
    
    csv_path = f"{RESULTS_DIR}exp3_class_importance.csv"
    columns = ['combo_name', 'classes', 'support_f1', 'deny_f1', 'query_f1', 'comment_f1', 'macro_f1']
    
    existing_df = load_or_create_csv(csv_path, columns)
    completed = set(existing_df.get('combo_name', []))
    
    all_classes = ['support', 'deny', 'query', 'comment']
    class_abbrev = {'support': 'S', 'deny': 'D', 'query': 'Q', 'comment': 'C'}
    
    # Generate all combinations: 1-shot singles, 2-shot pairs, 3-shot triples, 4-shot full
    combos = []
    for r in range(1, 5):
        for combo in combinations(all_classes, r):
            combos.append(list(combo))
    
    for classes in combos:
        combo_name = f"{len(classes)}-shot: " + "+".join(class_abbrev[c] for c in classes)
        
        if combo_name in completed:
            print(f"  Skipping {combo_name} (already done)")
            continue
        
        examples = get_few_shot_examples(train_df, n_per_class=1, use_set='diverse', classes=classes)
        metrics = evaluate_and_get_metrics(model, tokenizer, dev_df, mode="few-shot", examples=examples)
        
        row = {'combo_name': combo_name, 'classes': '+'.join(classes), **metrics}
        append_row_to_csv(csv_path, row)
        print(f"  {combo_name}: macro_f1={metrics['macro_f1']:.4f}")
    
    print(f"Results saved to {csv_path}")
    
    # Generate heatmap
    results_df = pd.read_csv(csv_path)
    heatmap_data = results_df.set_index('combo_name')[['support_f1', 'deny_f1', 'query_f1', 'comment_f1', 'macro_f1']]
    
    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax, vmin=0.15, vmax=0.65)
    ax.set_title('Class Importance in Few-shot Examples')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Few-shot Combination')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}exp3_class_importance.png", dpi=150)
    plt.close()


# =============================================================================
# EXPERIMENT 4: Test Set Final Comparison
# =============================================================================
def run_exp4_test_final(model, tokenizer, train_df, test_df):
    print("\n" + "="*60)
    print("EXPERIMENT 4: Test Set Final Comparison")
    print("="*60)
    
    csv_path = f"{RESULTS_DIR}exp4_test_final.csv"
    columns = ['strategy', 'repeat', 'macro_f1', 'support_f1', 'deny_f1', 'query_f1', 'comment_f1']
    
    existing_df = load_or_create_csv(csv_path, columns)
    completed = set(zip(existing_df.get('strategy', []), existing_df.get('repeat', [])))
    
    strategies = ['zero-shot', 'few-shot', 'cot', 'few-shot-cot']
    n_repeats = 3
    
    for strategy in strategies:
        for repeat in range(1, n_repeats + 1):
            if (strategy, repeat) in completed:
                print(f"  Skipping {strategy}/repeat={repeat} (already done)")
                continue
            
            # Configure examples based on strategy
            if strategy == 'zero-shot':
                examples = None
                mode = 'zero-shot'
            elif strategy == 'few-shot':
                examples = get_few_shot_examples(train_df, n_per_class=1, use_set='diverse')
                mode = 'few-shot'
            elif strategy == 'cot':
                examples = None
                mode = 'cot'
            elif strategy == 'few-shot-cot':
                examples = get_cot_examples(train_df)
                mode = 'cot'
            
            metrics = evaluate_and_get_metrics(model, tokenizer, test_df, mode=mode, examples=examples)
            
            row = {'strategy': strategy, 'repeat': repeat, **metrics}
            append_row_to_csv(csv_path, row)
            print(f"  {strategy}/repeat={repeat}: macro_f1={metrics['macro_f1']:.4f}")
    
    print(f"Results saved to {csv_path}")
    
    # Generate bar chart with averaged results
    results_df = pd.read_csv(csv_path)
    
    # Filter out classifier placeholder if present
    results_df = results_df[results_df['strategy'] != 'classifier']
    
    # Calculate means and stds
    summary = results_df.groupby('strategy')['macro_f1'].agg(['mean', 'std']).reset_index()
    summary.columns = ['strategy', 'mean_f1', 'std_f1']
    
    # Maintain order
    order = ['zero-shot', 'few-shot', 'cot', 'few-shot-cot']
    summary['strategy'] = pd.Categorical(summary['strategy'], categories=order, ordered=True)
    summary = summary.sort_values('strategy')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(summary['strategy'], summary['mean_f1'], yerr=summary['std_f1'], capsize=5)
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Macro F1 (mean ± std)')
    ax.set_title('Test Set Performance Comparison (3 repeats)')
    
    for bar, mean in zip(bars, summary['mean_f1']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{mean:.3f}', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}exp4_test_final.png", dpi=150)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
def generate_all_plots():
    """Generate all plots from saved CSVs (can run separately after experiments)."""
    print("Generating plots from saved CSVs...")
    
    # Exp 1 plots
    csv_path = f"{RESULTS_DIR}exp1_ablation.csv"
    if os.path.exists(csv_path):
        results_df = pd.read_csv(csv_path)
        for config_type in ['isolated', 'cumulative']:
            subset = results_df[results_df['config_type'] == config_type]
            if len(subset) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=subset, y='config_name', x='macro_f1', ax=ax)
                ax.set_xlabel('Macro F1')
                ax.set_ylabel('')
                ax.set_title(f'System Prompt Ablation: {config_type.title()}')
                plt.tight_layout()
                plt.savefig(f"{RESULTS_DIR}exp1_ablation_{config_type}.png", dpi=150)
                plt.close()
    
    # Exp 2 plot
    csv_path = f"{RESULTS_DIR}exp2_fewshot_strategies.csv"
    if os.path.exists(csv_path):
        results_df = pd.read_csv(csv_path)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=results_df, x='n_per_class', y='macro_f1', hue='strategy', ax=ax)
        ax.set_xlabel('Examples per Class')
        ax.set_ylabel('Macro F1')
        ax.set_title('Few-shot Strategy Comparison')
        ax.legend(title='Strategy')
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}exp2_fewshot_strategies.png", dpi=150)
        plt.close()
    
    # Exp 3 plot
    csv_path = f"{RESULTS_DIR}exp3_class_importance.csv"
    if os.path.exists(csv_path):
        results_df = pd.read_csv(csv_path)
        heatmap_data = results_df.set_index('combo_name')[['support_f1', 'deny_f1', 'query_f1', 'comment_f1', 'macro_f1']]
        fig, ax = plt.subplots(figsize=(10, 12))
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax, vmin=0.15, vmax=0.65)
        ax.set_title('Class Importance in Few-shot Examples')
        ax.set_xlabel('Metric')
        ax.set_ylabel('Few-shot Combination')
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}exp3_class_importance.png", dpi=150)
        plt.close()
    
    # Exp 4 plot
    csv_path = f"{RESULTS_DIR}exp4_test_final.csv"
    if os.path.exists(csv_path):
        results_df = pd.read_csv(csv_path)
        results_df = results_df[results_df['strategy'] != 'classifier']
        if len(results_df) > 0:
            summary = results_df.groupby('strategy')['macro_f1'].agg(['mean', 'std']).reset_index()
            summary.columns = ['strategy', 'mean_f1', 'std_f1']
            order = ['zero-shot', 'few-shot', 'cot', 'few-shot-cot']
            summary['strategy'] = pd.Categorical(summary['strategy'], categories=order, ordered=True)
            summary = summary.sort_values('strategy')
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(summary['strategy'], summary['mean_f1'], yerr=summary['std_f1'], capsize=5)
            ax.set_xlabel('Strategy')
            ax.set_ylabel('Macro F1 (mean ± std)')
            ax.set_title('Test Set Performance Comparison (3 repeats)')
            for bar, mean in zip(bars, summary['mean_f1']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{mean:.3f}', 
                        ha='center', va='bottom', fontsize=10)
            plt.tight_layout()
            plt.savefig(f"{RESULTS_DIR}exp4_test_final.png", dpi=150)
            plt.close()
    
    print("Plots saved.")


def run_all():
    """Run all experiments sequentially with single model load."""
    print("Starting Overnight Experiment Suite...")
    print(f"Results will be saved to: {RESULTS_DIR}")
    
    # Load data once
    print("Loading datasets...")
    train_df, dev_df, test_df = load_dataset()
    
    # Load model once (avoids CUDA OOM from multiple loads)
    print("Loading model (this may take a moment)...")
    model, tokenizer = load_model()
    
    # Run all experiments with shared model
    run_exp1_ablation(model, tokenizer, train_df, dev_df)
    run_exp2_fewshot_strategies(model, tokenizer, train_df, dev_df)
    run_exp3_class_importance(model, tokenizer, train_df, dev_df)
    run_exp4_test_final(model, tokenizer, train_df, test_df)
    
    # Generate all plots at the end
    generate_all_plots()
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    run_all()
