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
from transformers import pipeline

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

**Input Format:**
- [SRC] = The source tweet containing the rumour claim
- [PARENT] = The tweet that the target is directly replying to (if different from source)
- [TARGET] = The tweet whose stance you must classify

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

**Task:** Classify the stance of [TARGET] towards [SRC].
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
    """Create a text generation pipeline."""
    print(f"Creating pipeline for: {MODEL_NAME}")
    
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

def get_few_shot_examples(df, n_per_class=1):
    """Select stratified random examples for few-shot prompting."""
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


# %%
# Evaluation

def evaluate_prompting(pipe, df, mode="zero-shot", examples=None, batch_size=2, verbose=True):
    """Evaluate prompting on a dataset using batched inference. Increase batch_size if you have more VRAM."""
    
    # Prepare all messages upfront
    all_messages = []
    for _, row in df.iterrows():
        input_text = format_input_with_context(row, df, use_features=False)
        if mode == "zero-shot":
            messages = build_zero_shot_messages(input_text)
        else:
            messages = build_few_shot_messages(input_text, examples)
        all_messages.append(messages)
    
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
    few_shot_examples = get_few_shot_examples(train_df, n_per_class=1)
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

