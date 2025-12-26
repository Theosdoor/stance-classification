# %%
"""
SDQC Stance Classification via Prompting

Using open-source LLMs with zero-shot and few-shot prompting for
4-way stance classification: Support, Deny, Query, Comment

RumourEval 2017 Subtask A
"""

import os
import re
import json
import torch
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from transformers import pipeline
import seaborn as sns
import matplotlib.pyplot as plt

from data_loader import load_dataset, format_input_with_context

# %%
# Configuration

MODEL_OPTIONS = {
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "phi3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "gemma2-9b": "google/gemma-2-9b-it",
    "gemma3-12b": "google/gemma-3-12b-it",
}

DEFAULT_MODEL = "gemma3-12b"
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

def parse_stance_response(text):
    """Parse LLM response to extract stance label. Returns lowercase label or 'comment' as fallback."""
    text = text.upper().strip()
    
    for label in VALID_STANCES:
        if re.search(rf'\b{label}\b', text):
            return label.lower()
    
    return 'comment'  # Fallback


# %%
# Model loading

def create_pipeline(model_key=DEFAULT_MODEL):
    """Create a text generation pipeline."""
    model_name = MODEL_OPTIONS.get(model_key, model_key)
    print(f"Creating pipeline for: {model_name}")
    
    return pipeline(
        "text-generation",
        model=model_name,
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
    label = parse_stance_response(response)
    return label, response


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
    from datasets import Dataset
    
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
    
    # Parse responses
    predictions = [parse_stance_response(r) for r in raw_responses]
    true_labels = df['label_text'].tolist()
    
    # Metrics
    labels = ['support', 'deny', 'query', 'comment']
    macro_f1 = f1_score(true_labels, predictions, average='macro')
    per_class_f1 = f1_score(true_labels, predictions, average=None, labels=labels)
    
    if verbose:
        print(f"\n{'='*60}\nResults ({mode})\n{'='*60}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"\nPer-class F1:")
        for lbl, f1 in zip(labels, per_class_f1):
            print(f"  {lbl}: {f1:.4f}")
        print(f"\n{classification_report(true_labels, predictions, labels=labels)}")
    
    return {
        'predictions': predictions,
        'true_labels': true_labels,
        'raw_responses': raw_responses,
        'macro_f1': macro_f1,
        'per_class_f1': dict(zip(labels, per_class_f1)),
        'confusion_matrix': confusion_matrix(true_labels, predictions, labels=labels)
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


def save_results(results, model_key, mode):
    """Save evaluation results to JSON."""
    output = {
        'model': model_key,
        'mode': mode,
        'macro_f1': results['macro_f1'],
        'per_class_f1': results['per_class_f1'],
        'predictions': results['predictions'],
        'true_labels': results['true_labels'],
        'raw_responses': results['raw_responses']
    }
    
    filename = f"{RESULTS_DIR}{model_key}_{mode}_results.json"
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {filename}")


# %%
# Main experiment

def run_experiment(model_key=DEFAULT_MODEL, test_on="test", save=True):
    """Run full zero-shot and few-shot evaluation experiment."""
    print(f"\n{'='*60}")
    print(f"SDQC Prompting Experiment - {model_key}")
    print(f"{'='*60}\n")
    
    # Load data
    train_df, dev_df, test_df = load_dataset()
    eval_df = test_df if test_on == "test" else dev_df
    print(f"Train: {len(train_df)}, Eval ({test_on}): {len(eval_df)}")
    
    # Load pipeline
    pipe = create_pipeline(model_key)
    
    # Few-shot examples
    few_shot_examples = get_few_shot_examples(train_df, n_per_class=1)
    print(f"Few-shot examples: {len(few_shot_examples)}")
    
    # Zero-shot
    print(f"\n{'='*60}\nZero-Shot Evaluation\n{'='*60}")
    zero_results = evaluate_prompting(pipe, eval_df, mode="zero-shot")
    
    if save:
        save_results(zero_results, model_key, "zero-shot")
        plot_confusion_matrix(zero_results['confusion_matrix'], "Zero-Shot",
                              f"{RESULTS_DIR}{model_key}_zero_shot_cm.png")
    
    # Few-shot
    print(f"\n{'='*60}\nFew-Shot Evaluation\n{'='*60}")
    few_results = evaluate_prompting(pipe, eval_df, mode="few-shot", examples=few_shot_examples)
    
    if save:
        save_results(few_results, model_key, "few-shot")
        plot_confusion_matrix(few_results['confusion_matrix'], "Few-Shot",
                              f"{RESULTS_DIR}{model_key}_few_shot_cm.png")
    
    # Summary
    print(f"\n{'='*60}\nSummary\n{'='*60}")
    print(f"Zero-Shot Macro F1: {zero_results['macro_f1']:.4f}")
    print(f"Few-Shot Macro F1:  {few_results['macro_f1']:.4f}")
    print(f"Improvement: {(few_results['macro_f1'] - zero_results['macro_f1'])*100:.2f}%")
    
    return {'zero_shot': zero_results, 'few_shot': few_results}


# %%
# CLI

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SDQC classification via prompting")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODEL_OPTIONS.keys()))
    parser.add_argument("--test-on", default="test", choices=["test", "dev"])
    parser.add_argument("--no-save", action="store_true")
    
    args = parser.parse_args()
    run_experiment(model_key=args.model, test_on=args.test_on, save=not args.no_save)
