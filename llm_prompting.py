# %%
"""
SDQC Stance Classification via Prompting

Using open-source LLMs with zero-shot and few-shot prompting for
4-way stance classification: Support, Deny, Query, Comment

RumourEval 2017 Subtask A
"""

import argparse
import gc
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tqdm.auto import tqdm
from transformers import pipeline, AutoTokenizer

from data_loader import load_dataset, format_input_with_context

# %%
# Configuration

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
RESULTS_DIR = "./results/prompting/"
SEED = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
np.random.seed(SEED)

# %%
# Prompts

SYS_PROMPT = """\
You are an expert in rumour stance analysis on social media.

Your task is to classify the stance of the TARGET tweet towards veracity of the rumour in the SOURCE tweet.

**Input Structure:**
The input will be provided as a single string containing labeled segments:
"[SRC] ... [PARENT] ... [TARGET] ..." (Note: [PARENT] may be omitted if not applicable).

**Classification Labels:**
- **SUPPORT**: The reply supports the veracity of the source claim
- **DENY**: The reply denies the veracity of the source claim
- **QUERY**: The reply asks for additional evidence in relation to the veracity of the source claim
- **COMMENT**: The reply makes their own comment without a clear contribution to assessing the veracity of the source claim

**Output Format:**
Respond with ONLY one word: SUPPORT, DENY, QUERY, or COMMENT
"""

USER_PROMPT_TEMPLATE = """\
**Thread Context:**
{thread_context}

**Task:** Classify the stance of [TARGET] towards the veracity of the rumour in [SRC].
"""

# %%
# Message builders

def build_user_prompt(thread_context):
    """Build the user prompt for stance classification."""
    return USER_PROMPT_TEMPLATE.format(thread_context=thread_context)


def build_zero_shot_messages(thread_context):
    """Build zero-shot messages for stance classification."""
    return [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": build_user_prompt(thread_context)},
    ]


def build_few_shot_messages(thread_context, examples=None):
    """Build few-shot messages with examples as user/assistant turns."""
    messages = [{"role": "system", "content": SYS_PROMPT}]
    
    if examples:
        for ex in examples:
            messages.append({"role": "user", "content": build_user_prompt(ex['source'])})
            messages.append({"role": "assistant", "content": ex['label']})
    
    messages.append({"role": "user", "content": build_user_prompt(thread_context)})
    return messages


# %%
# Response parsing

VALID_STANCES = ['SUPPORT', 'DENY', 'QUERY', 'COMMENT']
REFUSAL_PATTERNS = ['CANNOT', 'UNABLE', 'I\'M NOT', 'I AM NOT', 'SORRY']

def parse_stance_response(text):
    """
    Parse LLM response to extract stance label.
    Returns tuple: (label, error_type)
    - label: lowercase stance label or None if parsing fails
    - error_type: None if success, else 'empty'|'refusal'|'multiple'|'no_label'
    """
    if not text or not text.strip():
        return None, 'empty'
    
    text_upper = text.upper().strip()
    
    # Check for refusal
    if any(pat in text_upper for pat in REFUSAL_PATTERNS):
        return None, 'refusal'
    
    # Find all matching labels
    found = [label for label in VALID_STANCES if re.search(rf'\b{label}\b', text_upper)]
    
    if len(found) == 1:
        return found[0].lower(), None  # Success
    elif len(found) > 1:
        return None, 'multiple'  # Ambiguous
    else:
        return None, 'no_label'  # No valid stance found


# %%
# Model loading

def create_pipeline():    
    return pipeline(
        "text-generation",
        model=MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )


# %%
# Inference

def generate_response(pipe, messages, max_new_tokens=10):
    """Generate a response from the pipeline."""
    output = pipe(messages, max_new_tokens=max_new_tokens)
    return output[0]["generated_text"][-1]["content"].strip()


def classify_instance(pipe, input_text, mode="zero-shot", examples=None):
    """Classify a single instance. Returns (predicted_label, raw_response)."""
    if mode == "zero-shot":
        messages = build_zero_shot_messages(input_text)
    else:
        messages = build_few_shot_messages(input_text, examples)
    
    response = generate_response(pipe, messages)
    label, error = parse_stance_response(response)
    return label, response, error


# %%
# Few-shot example selection

def get_random_stratified_few_shot_examples(df, n_per_class=1):
    """Select stratified random examples for few-shot prompting (non-deterministic)."""
    examples = []
    
    for label in ['support', 'deny', 'query', 'comment']:
        class_df = df[df['label_text'] == label]
        samples = class_df.sample(n=min(n_per_class, len(class_df)))
        
        for _, row in samples.iterrows():
            examples.append({
                'source': format_input_with_context(row, df, use_features=False),
                'label': label.upper()
            })
    
    return examples


def get_same_src_few_shot_examples(df, n_per_class=1, source_id=None):
    """
    Select few-shot examples all from the same source tweet thread.
    
    This ensures all examples share the same rumour context, which may help
    the model better understand stance classification within a single thread.
    
    Args:
        df: DataFrame with tweet data
        n_per_class: Number of examples per stance class
        source_id: Optional specific source ID to use. If None, finds a thread
                   with all 4 stances automatically.
    
    Returns:
        List of example dicts with 'source' (formatted input) and 'label'
    """
    all_four = {'support', 'deny', 'query', 'comment'}
    
    # Find a source with all 4 stances if not specified
    if source_id is None:
        candidates = []
        for src_id in df['source_id'].unique():
            replies = df[(df['source_id'] == src_id) & (df['tweet_id'] != src_id)]
            if len(replies) > 0:
                stances = set(replies['label_text'].unique())
                if stances == all_four:
                    # Score by balance (prefer equal distribution)
                    counts = replies['label_text'].value_counts()
                    balance_score = counts.min() / counts.max()
                    candidates.append((src_id, balance_score, len(replies)))
        
        if not candidates:
            print("Warning: No source has all 4 stances, falling back to random selection")
            return get_random_stratified_few_shot_examples(df, n_per_class)
        
        # Sort by balance score (most balanced first), then by size (smaller preferred)
        candidates.sort(key=lambda x: (-x[1], x[2]))
        source_id = candidates[0][0]
        print(f"Using source {source_id} for few-shot examples (balance: {candidates[0][1]:.2f})")
    
    # Get replies from this source
    thread_df = df[(df['source_id'] == source_id) & (df['tweet_id'] != source_id)]
    
    examples = []
    for label in ['support', 'deny', 'query', 'comment']:
        class_df = thread_df[thread_df['label_text'] == label]
        if len(class_df) == 0:
            print(f"Warning: No {label} examples in thread {source_id}")
            continue
        samples = class_df.sample(n=min(n_per_class, len(class_df)))
        
        for _, row in samples.iterrows():
            examples.append({
                'source': format_input_with_context(row, df, use_features=False),
                'label': label.upper()
            })
    
    return examples


# Deterministic few-shot examples from diverse sources/topics
# Selected manually to cover different rumour types and topics
DIVERSE_FEW_SHOT_IDS = {
    # Each from a different topic for maximum diversity (TRAINING DATA ONLY)
    'support': '553486439129038848',   # charliehebdo - clear support
    'deny': '500384255814299648',       # ferguson - explicit denial
    'query': '525003827494531073',      # ottawashooting - questioning veracity  
    'comment': '544314008980172800',    # sydneysiege - neutral observation
}


# Deterministic few-shot examples from same source thread (TRAINING DATA ONLY)
# Source 529739968470867968 from 'prince-toronto' topic - has all 4 stances, most balanced
SAME_SRC_SOURCE_ID = '529739968470867968'
SAME_SRC_FEW_SHOT_IDS = {
    'support': '529740822238232577',   # support reply in this thread
    'deny': '529740809672077313',       # deny reply in this thread
    'query': '529748991849013248',      # query reply in this thread
    'comment': '529741574742507520',    # comment reply in this thread
}


def get_diverse_few_shot_examples(df):
    """
    Get deterministic few-shot examples from different sources and topics.
    
    Uses pre-selected tweet IDs that represent clear examples of each stance
    from diverse rumour topics. This ensures reproducibility for coursework.
    
    Returns:
        List of example dicts with 'source' (formatted input) and 'label'
    """
    examples = []
    
    for label, tweet_id in DIVERSE_FEW_SHOT_IDS.items():
        row = df[df['tweet_id'] == tweet_id]
        if len(row) == 0:
            print(f"Warning: Tweet {tweet_id} not found in dataset for {label}")
            continue
        row = row.iloc[0]
        examples.append({
            'source': format_input_with_context(row, df, use_features=False),
            'label': label.upper()
        })
    
    return examples


def get_same_src_few_shot_examples_deterministic(df):
    """
    Get deterministic few-shot examples all from the same source thread.
    
    Uses pre-selected tweet IDs from a single thread with all 4 stances.
    This ensures reproducibility for coursework.
    
    Returns:
        List of example dicts with 'source' (formatted input) and 'label'
    """
    examples = []
    
    for label, tweet_id in SAME_SRC_FEW_SHOT_IDS.items():
        row = df[df['tweet_id'] == tweet_id]
        if len(row) == 0:
            print(f"Warning: Tweet {tweet_id} not found in dataset for {label}")
            continue
        row = row.iloc[0]
        examples.append({
            'source': format_input_with_context(row, df, use_features=False),
            'label': label.upper()
        })
    
    return examples


# Deterministic random-stratified examples (fixed seed selection from training)
# Pre-selected to simulate random stratified sampling, reproducible
RANDOM_STRATIFIED_FEW_SHOT_IDS = {
    'support': '544306719686656000',   # sydneysiege
    'deny': '524990163446140928',       # ottawashooting
    'query': '500386447158161408',      # ferguson
    'comment': '553543395717550080',    # charliehebdo
}


def get_random_stratified_few_shot_examples_deterministic(df):
    """
    Get deterministic 'random' stratified examples for reproducibility.
    
    Uses pre-selected tweet IDs that simulate random stratified sampling.
    Each example is from a different source tweet.
    
    Returns:
        List of example dicts with 'source' (formatted input) and 'label'
    """
    examples = []
    
    for label, tweet_id in RANDOM_STRATIFIED_FEW_SHOT_IDS.items():
        row = df[df['tweet_id'] == tweet_id]
        if len(row) == 0:
            print(f"Warning: Tweet {tweet_id} not found in dataset for {label}")
            continue
        row = row.iloc[0]
        examples.append({
            'source': format_input_with_context(row, df, use_features=False),
            'label': label.upper()
        })
    
    return examples


# %%
# Strategy Comparison Experiment

FEW_SHOT_STRATEGIES = {
    'diverse': get_diverse_few_shot_examples,
    'same_src': get_same_src_few_shot_examples_deterministic,
    'random_stratified': get_random_stratified_few_shot_examples_deterministic,
}


def run_strategy_comparison(pipe, train_df, eval_df, save=True):
    """
    Compare few-shot selection strategies.
    
    Runs evaluation with each strategy and returns results for visualization.
    All strategies use deterministic pre-selected examples for reproducibility.
    """
    results = {}
    
    # Zero-shot baseline
    print(f"\n{'='*60}\nZero-Shot Baseline\n{'='*60}")
    results['zero_shot'] = evaluate_prompting(pipe, eval_df, mode="zero-shot", verbose=True)
    
    # Each strategy
    for strategy_name, get_examples_fn in FEW_SHOT_STRATEGIES.items():
        print(f"\n{'='*60}\n{strategy_name} Strategy\n{'='*60}")
        examples = get_examples_fn(train_df)
        print(f"  Using {len(examples)} examples")
        results[strategy_name] = evaluate_prompting(
            pipe, eval_df, mode="few-shot", examples=examples, verbose=True
        )
    
    if save:
        # Save results
        output = {
            strategy: {
                'macro_f1': r['macro_f1'],
                'per_class_f1': r['per_class_f1'],
                'error_rate': r['error_rate']
            }
            for strategy, r in results.items()
        }
        with open(f"{RESULTS_DIR}strategy_comparison.json", 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved: {RESULTS_DIR}strategy_comparison.json")
    
    return results


def plot_strategy_comparison(results, save_path=None):
    """
    Create grouped bar chart comparing few-shot strategies.
    
    X-axis: Strategies (zero_shot, diverse, same_src, random_stratified)
    Y-axis: F1 Score
    Bars: Macro F1 and per-class F1
    """
    strategies = ['zero_shot', 'diverse', 'same_src', 'random_stratified']
    strategy_labels = ['Zero-Shot', 'Diverse', 'Same Source', 'Random Stratified']
    metrics = ['macro_f1', 'support', 'deny', 'query', 'comment']
    metric_labels = ['Macro F1', 'Support', 'Deny', 'Query', 'Comment']
    
    # Prepare data
    data = []
    for strategy in strategies:
        r = results[strategy]
        row = [r['macro_f1']]
        for cls in ['support', 'deny', 'query', 'comment']:
            row.append(r['per_class_f1'][cls])
        data.append(row)
    
    data = np.array(data)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(strategies))
    width = 0.15
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
    
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        offset = (i - 2) * width
        bars = ax.bar(x + offset, data[:, i], width, label=label, color=color, alpha=0.85)
        # Add value labels on bars
        for bar, val in zip(bars, data[:, i]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Few-Shot Strategy', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Few-Shot Strategy Comparison: Effect on Stance Classification', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(strategy_labels)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# %%
# Evaluation

def evaluate_prompting(pipe, df, mode="zero-shot", examples=None, batch_size=2, verbose=True):
    """Evaluate prompting on a dataset using batched inference. Increase batch_size if you have more VRAM."""
    
    # Load tokenizer for token counting
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Prepare all messages upfront
    all_messages = []
    token_counts = []
    for _, row in df.iterrows():
        input_text = format_input_with_context(row, df, use_features=False)
        if mode == "zero-shot":
            messages = build_zero_shot_messages(input_text)
        else:
            messages = build_few_shot_messages(input_text, examples)
        all_messages.append(messages)
        
        # Count tokens using chat template
        prompt_tokens = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        token_counts.append(prompt_tokens)
    
    # Log token statistics
    if verbose:
        print(f"\nToken Statistics ({mode}):")
        print(f"  Min: {min(token_counts)}, Max: {max(token_counts)}")
        print(f"  Mean: {np.mean(token_counts):.1f}, Median: {np.median(token_counts):.1f}")
        if max(token_counts) > 7500:
            print(f"  ⚠️  WARNING: {sum(1 for t in token_counts if t > 7500)} prompts exceed 7500 tokens!")
        else:
            print(f"  ✓ All prompts within 8K context limit")
    
    # Batched inference using Dataset
    dataset = Dataset.from_dict({"messages": all_messages})
    
    raw_responses = []
    for out in tqdm(pipe(dataset["messages"], max_new_tokens=10, batch_size=batch_size), 
                    total=len(df), desc=f"Evaluating ({mode})"):
        raw_responses.append(out[0]["generated_text"][-1]["content"].strip())
    
    # Parse responses and track errors
    parsed_results = [parse_stance_response(r) for r in raw_responses]
    labels_parsed = [p[0] for p in parsed_results]
    error_types = [p[1] for p in parsed_results]
    true_labels = df['label_text'].tolist()
    
    # Count error types
    error_counts = Counter(e for e in error_types if e is not None)
    total_errors = sum(error_counts.values())
    error_rate = total_errors / len(parsed_results)
    
    # For metrics, filter out parse errors
    valid_indices = [i for i, label in enumerate(labels_parsed) if label is not None]
    predictions = [labels_parsed[i] for i in valid_indices]
    filtered_true = [true_labels[i] for i in valid_indices]
    
    # Metrics (on valid predictions only)
    stance_labels = ['support', 'deny', 'query', 'comment']
    if len(predictions) > 0:
        macro_f1 = f1_score(filtered_true, predictions, average='macro')
        per_class_f1 = f1_score(filtered_true, predictions, average=None, labels=stance_labels)
    else:
        macro_f1 = 0.0
        per_class_f1 = [0.0, 0.0, 0.0, 0.0]
    
    if verbose:
        print(f"\n{'='*60}\nResults ({mode})\n{'='*60}")
        print(f"Parse errors: {total_errors}/{len(parsed_results)} ({error_rate:.1%})")
        if error_counts:
            print(f"  Error breakdown: {dict(error_counts)}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"\nPer-class F1:")
        for lbl, f1 in zip(stance_labels, per_class_f1):
            print(f"  {lbl}: {f1:.4f}")
        if len(predictions) > 0:
            print(f"\n{classification_report(filtered_true, predictions, labels=stance_labels)}")
    
    return {
        'predictions': labels_parsed,  # includes None for errors
        'true_labels': true_labels,
        'raw_responses': raw_responses,
        'error_types': error_types,
        'macro_f1': macro_f1,
        'per_class_f1': dict(zip(stance_labels, per_class_f1)),
        'confusion_matrix': confusion_matrix(filtered_true, predictions, labels=stance_labels) if len(predictions) > 0 else None,
        'error_counts': dict(error_counts),
        'total_errors': total_errors,
        'error_rate': error_rate,
    }


def plot_confusion_matrix(cm, mode, save_path=None):
    """Plot confusion matrix."""
    labels = ['support', 'deny', 'query', 'comment']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix ({mode})')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def save_results(results, mode):
    """Save evaluation results to JSON."""
    output = {
        'model': MODEL_NAME,
        'mode': mode,
        'macro_f1': results['macro_f1'],
        'per_class_f1': results['per_class_f1'],
        'predictions': results['predictions'],
        'true_labels': results['true_labels'],
        'raw_responses': results['raw_responses']
    }
    
    filename = f"{RESULTS_DIR}llama3.1-8b_{mode}_results.json"
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {filename}")


# %%
# Main experiment

def run_experiment(test_on="test", save=True):
    """Run full zero-shot and few-shot evaluation experiment."""
    print(f"\n{'='*60}")
    print(f"SDQC Prompting Experiment - {MODEL_NAME}")
    print(f"{'='*60}\n")
    
    # Load data
    train_df, dev_df, test_df = load_dataset()
    eval_df = test_df if test_on == "test" else dev_df
    print(f"Train: {len(train_df)}, Eval ({test_on}): {len(eval_df)}")
    
    # Load pipeline
    pipe = create_pipeline()
    
    # Few-shot examples
    few_shot_examples = get_diverse_few_shot_examples(train_df)
    print(f"Few-shot examples: {len(few_shot_examples)}")
    
    # Zero-shot
    print(f"\n{'='*60}\nZero-Shot Evaluation\n{'='*60}")
    zero_results = evaluate_prompting(pipe, eval_df, mode="zero-shot")
    
    if save:
        save_results(zero_results, "zero-shot")
        plot_confusion_matrix(zero_results['confusion_matrix'], "Zero-Shot",
                              f"{RESULTS_DIR}llama3.1-8b_zero_shot_cm.png")
    
    # Few-shot
    print(f"\n{'='*60}\nFew-Shot Evaluation\n{'='*60}")
    few_results = evaluate_prompting(pipe, eval_df, mode="few-shot", examples=few_shot_examples)
    
    if save:
        save_results(few_results, "few-shot")
        plot_confusion_matrix(few_results['confusion_matrix'], "Few-Shot",
                              f"{RESULTS_DIR}llama3.1-8b_few_shot_cm.png")
    
    # Summary
    print(f"\n{'='*60}\nSummary\n{'='*60}")
    print(f"Zero-Shot Macro F1: {zero_results['macro_f1']:.4f}")
    print(f"Few-Shot Macro F1:  {few_results['macro_f1']:.4f}")
    print(f"Improvement: {(few_results['macro_f1'] - zero_results['macro_f1'])*100:.2f}%")
    
    return {'zero_shot': zero_results, 'few_shot': few_results}


# %%
# CLI

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="SDQC classification via prompting")
    parser.add_argument("--test-on", default="test", choices=["test", "dev"])
    parser.add_argument("--no-save", action="store_true")
    
    args = parser.parse_args()
    
    run_experiment(test_on=args.test_on, save=not args.no_save)

