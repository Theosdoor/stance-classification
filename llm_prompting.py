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
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import seaborn as sns
import matplotlib.pyplot as plt

from data_loader import (
    load_dataset, format_input_with_context,
    LABEL_TO_ID, ID_TO_LABEL
)

# ============================================================================
# Configuration
# ============================================================================

# Model options - open-source instruction-tuned LLMs
MODEL_OPTIONS = {
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "phi3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "gemma2-9b": "google/gemma-2-9b-it",
}

# Default model
DEFAULT_MODEL = "llama3.1-8b"

# Device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)

# Results directory
RESULTS_DIR = "./results/prompting/"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Random seed
SEED = 42

# ============================================================================
# Prompt Templates
# ============================================================================

# Note: Prompts now built dynamically in build_zero_shot_prompt() and build_few_shot_prompt()
# to properly format thread context with [SRC]/[PARENT]/[TARGET] markers

# Few-shot examples - Now dynamically selected from training data
# Strategy: prefer depth > 1 (has context), diverse topics
# See get_random_examples() for implementation

def build_few_shot_prompt(thread_context: str, _unused: str = '', examples: list = None) -> str:
    """
    Build a few-shot prompt with examples followed by the test instance.
    
    Args:
        thread_context: Formatted thread context from format_input_with_context
        _unused: Kept for API compatibility
        examples: List of example dicts with 'source', 'reply', 'label' keys
    
    Returns:
        Formatted few-shot prompt string
    """
    prompt = """You are an expert in rumour stance analysis on social media.

**Input Format:**
- [SRC] = The source tweet containing the rumour claim
- [PARENT] = The tweet that the target is directly replying to (if different from source)
- [TARGET] = The tweet whose stance you must classify

**Classification Categories:**
- SUPPORT: The target agrees with or supports the source claim
- DENY: The target disagrees with or refutes the source claim
- QUERY: The target questions or seeks verification of the source claim
- COMMENT: The target makes a neutral comment without taking a stance

**Examples:**
"""
    
    if examples:
        for ex in examples:
            prompt += f"""
---
Thread: {ex['source']}
Classification: {ex['label']}
"""
    
    prompt += f"""
---

Now classify the following:

**Thread Context:**
{thread_context}

**Task:** Classify the stance of [TARGET] towards [SRC].

Classification:"""
    
    return prompt


def build_zero_shot_prompt(thread_context: str, _unused: str = '') -> str:
    """Build a zero-shot prompt for stance classification."""
    return f"""You are an expert in rumour stance analysis on social media.

**Input Format:**
- [SRC] = The source tweet containing the rumour claim
- [PARENT] = The tweet that the target is directly replying to (if different from source)
- [TARGET] = The tweet whose stance you must classify

**Thread Context:**
{thread_context}

**Classification Task:**
Classify the stance of [TARGET] towards [SRC] into exactly ONE of these categories:
- **SUPPORT**: The target agrees with or supports the source claim
- **DENY**: The target disagrees with or refutes the source claim
- **QUERY**: The target questions or seeks verification of the source claim
- **COMMENT**: The target makes a neutral comment without taking a stance

Respond with ONLY one word: SUPPORT, DENY, QUERY, or COMMENT"""


# ============================================================================
# Response Parsing
# ============================================================================

def parse_stance_response(text: str) -> str:
    """
    Parse LLM response to extract stance label.
    
    Handles various response formats:
    - Direct label: "SUPPORT"
    - With explanation: "The classification is SUPPORT because..."
    - Lowercase: "support"
    
    Returns:
        Lowercase stance label or 'comment' as fallback
    """
    text = text.upper().strip()
    
    # Check for each label in order of specificity
    # (COMMENT last as it's most common and sometimes appears in explanations)
    label_order = ['SUPPORT', 'DENY', 'QUERY', 'COMMENT']
    
    for label in label_order:
        # Match standalone label or at start of response
        if re.search(rf'\b{label}\b', text):
            return label.lower()
    
    # Fallback to comment (majority class)
    return 'comment'


# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_key: str = DEFAULT_MODEL):
    """
    Load a HuggingFace model and tokenizer.
    
    Args:
        model_key: Key from MODEL_OPTIONS dict
    
    Returns:
        Tuple of (model, tokenizer)
    """
    model_name = MODEL_OPTIONS.get(model_key, model_key)
    print(f"Loading model: {model_name}")
    print(f"Device: {DEVICE}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with appropriate settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if DEVICE.type in ['cuda', 'mps'] else torch.float32,
        device_map="auto",
        trust_remote_code=True  # Required for some models like Qwen
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def create_pipeline(model_key: str = DEFAULT_MODEL):
    """
    Create a text generation pipeline for easier inference.
    
    Args:
        model_key: Key from MODEL_OPTIONS dict
    
    Returns:
        HuggingFace pipeline
    """
    model_name = MODEL_OPTIONS.get(model_key, model_key)
    print(f"Creating pipeline for: {model_name}")
    
    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.float16 if DEVICE.type in ['cuda', 'mps'] else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    return pipe


# ============================================================================
# Inference
# ============================================================================

def generate_response(
    model, 
    tokenizer, 
    prompt: str, 
    max_new_tokens: int = 20
) -> str:
    """
    Generate a response from the model.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Generated text (response only, not including prompt)
    """
    # Format as chat if model supports it
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for reproducibility
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only new tokens
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    return response.strip()


def classify_instance(
    model, 
    tokenizer, 
    input_text: str, 
    mode: str = "zero-shot",
    examples: list = None
) -> tuple:
    """
    Classify a single instance.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        input_text: Formatted input text (from format_input_with_context)
        mode: "zero-shot" or "few-shot"
        examples: Custom few-shot examples (optional)
    
    Returns:
        Tuple of (predicted_label, raw_response)
    """
    if mode == "zero-shot":
        prompt = build_zero_shot_prompt(input_text, '')
    else:
        prompt = build_few_shot_prompt(input_text, '', examples)
    
    response = generate_response(model, tokenizer, prompt)
    label = parse_stance_response(response)
    
    return label, response


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_prompting(
    model, 
    tokenizer, 
    df: pd.DataFrame, 
    mode: str = "zero-shot",
    examples: list = None,
    verbose: bool = True
) -> dict:
    """
    Evaluate prompting approach on a dataset.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        df: DataFrame with 'source_text', 'reply_text', 'label_text' columns
        mode: "zero-shot" or "few-shot"
        examples: Custom few-shot examples (optional)
        verbose: Whether to print progress
    
    Returns:
        Dict with predictions, labels, metrics, and raw responses
    """
    predictions = []
    true_labels = []
    raw_responses = []
    
    iterator = tqdm(df.iterrows(), total=len(df), desc=f"Evaluating ({mode})")
    
    for idx, row in iterator:
        input_text = format_input_with_context(row, df, use_features=False)
        pred, response = classify_instance(
            model, tokenizer,
            input_text,
            mode=mode, examples=examples
        )
        
        predictions.append(pred)
        true_labels.append(row['label_text'])
        raw_responses.append(response)
    
    # Compute metrics
    macro_f1 = f1_score(true_labels, predictions, average='macro')
    per_class_f1 = f1_score(true_labels, predictions, average=None, 
                            labels=['support', 'deny', 'query', 'comment'])
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Results ({mode})")
        print(f"{'='*60}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"\nPer-class F1:")
        for label, f1 in zip(['support', 'deny', 'query', 'comment'], per_class_f1):
            print(f"  {label}: {f1:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(true_labels, predictions, 
                                    labels=['support', 'deny', 'query', 'comment']))
    
    return {
        'predictions': predictions,
        'true_labels': true_labels,
        'raw_responses': raw_responses,
        'macro_f1': macro_f1,
        'per_class_f1': dict(zip(['support', 'deny', 'query', 'comment'], per_class_f1)),
        'confusion_matrix': confusion_matrix(true_labels, predictions, 
                                             labels=['support', 'deny', 'query', 'comment'])
    }


def plot_confusion_matrix(cm: np.ndarray, mode: str, save_path: str = None):
    """Plot and optionally save confusion matrix."""
    labels = ['support', 'deny', 'query', 'comment']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix ({mode})')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def save_results(results: dict, model_key: str, mode: str):
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
    print(f"Saved results to: {filename}")


# ============================================================================
# Few-Shot Example Selection Strategies
# ============================================================================

def get_random_examples(df: pd.DataFrame, n_per_class: int = 1, seed: int = SEED) -> list:
    """
    Select random stratified examples for few-shot prompting.
    
    Args:
        df: Training DataFrame
        n_per_class: Number of examples per class
        seed: Random seed
    
    Returns:
        List of example dicts
    """
    np.random.seed(seed)
    examples = []
    
    for label in ['support', 'deny', 'query', 'comment']:
        class_df = df[df['label_text'] == label]
        samples = class_df.sample(n=min(n_per_class, len(class_df)))
        
        for _, row in samples.iterrows():
            examples.append({
                'source': format_input_with_context(row, df, use_features=False),
                'reply': '',
                'label': label.upper()
            })
    
    return examples


def get_diverse_examples(df: pd.DataFrame, n_per_class: int = 1) -> list:
    """
    Select diverse examples by picking from different topics.
    
    Args:
        df: Training DataFrame
        n_per_class: Number of examples per class
    
    Returns:
        List of example dicts
    """
    examples = []
    used_topics = set()
    
    for label in ['support', 'deny', 'query', 'comment']:
        class_df = df[df['label_text'] == label]
        
        # Try to pick from unused topics
        for _, row in class_df.iterrows():
            if row['topic'] not in used_topics or len(examples) % 4 == 3:
                examples.append({
                    'source': format_input_with_context(row, class_df, use_features=False),
                    'reply': '',
                    'label': label.upper()
                })
                used_topics.add(row['topic'])
                break
        
        # Fallback to first available
        if len(examples) % 4 != 0 or len(examples) == 0:
            row = class_df.iloc[0]
            examples.append({
                'source': format_input_with_context(row, class_df, use_features=False),
                'reply': '',
                'label': label.upper()
            })
    
    return examples[:4]  # Ensure exactly 4


# ============================================================================
# Main Experiments
# ============================================================================

def run_experiment(
    model_key: str = DEFAULT_MODEL,
    test_on: str = "test",
    save: bool = True
):
    """
    Run full evaluation experiment.
    
    Args:
        model_key: Model to use from MODEL_OPTIONS
        test_on: "test" or "dev" split
        save: Whether to save results
    """
    print(f"\n{'='*60}")
    print(f"SDQC Classification Prompting Experiment")
    print(f"Model: {model_key}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading data...")
    train_df, dev_df, test_df = load_dataset()
    eval_df = test_df if test_on == "test" else dev_df
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(eval_df)}")
    
    # Load model
    model, tokenizer = load_model(model_key)
    
    # Get few-shot examples from training data
    few_shot_examples = get_random_examples(train_df, n_per_class=1)
    print(f"\nFew-shot examples:")
    for ex in few_shot_examples:
        print(f"  {ex['label']}: {ex['reply'][:50]}...")
    
    # Zero-shot evaluation
    print(f"\n{'='*60}")
    print("Zero-Shot Evaluation")
    print(f"{'='*60}")
    zero_shot_results = evaluate_prompting(
        model, tokenizer, eval_df, 
        mode="zero-shot"
    )
    
    if save:
        save_results(zero_shot_results, model_key, "zero-shot")
        plot_confusion_matrix(
            zero_shot_results['confusion_matrix'], 
            "Zero-Shot",
            f"{RESULTS_DIR}{model_key}_zero_shot_cm.png"
        )
    
    # Few-shot evaluation
    print(f"\n{'='*60}")
    print("Few-Shot Evaluation")
    print(f"{'='*60}")
    few_shot_results = evaluate_prompting(
        model, tokenizer, eval_df,
        mode="few-shot",
        examples=few_shot_examples
    )
    
    if save:
        save_results(few_shot_results, model_key, "few-shot")
        plot_confusion_matrix(
            few_shot_results['confusion_matrix'],
            "Few-Shot",
            f"{RESULTS_DIR}{model_key}_few_shot_cm.png"
        )
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("Summary Comparison")
    print(f"{'='*60}")
    print(f"Zero-Shot Macro F1: {zero_shot_results['macro_f1']:.4f}")
    print(f"Few-Shot Macro F1:  {few_shot_results['macro_f1']:.4f}")
    print(f"Improvement: {(few_shot_results['macro_f1'] - zero_shot_results['macro_f1'])*100:.2f}%")
    
    return {
        'zero_shot': zero_shot_results,
        'few_shot': few_shot_results
    }


# ============================================================================
# Quick Test (for debugging prompts)
# ============================================================================

def quick_test():
    """Test prompts on a few examples without loading model."""
    print("Zero-Shot Prompt Example:")
    print("="*60)
    print(build_zero_shot_prompt(
        "Breaking: Fire reported at downtown building",
        "I can see smoke from my window, this is real"
    ))
    print("\n")
    
    print("Few-Shot Prompt Example:")
    print("="*60)
    print(build_few_shot_prompt(
        "Breaking: Fire reported at downtown building",
        "I can see smoke from my window, this is real"
    ))
    print("\n")
    
    # Test parsing
    print("Response Parsing Tests:")
    print("="*60)
    test_responses = [
        "SUPPORT",
        "The classification is DENY because the reply contradicts...",
        "query",
        "Based on the content, I would classify this as COMMENT.",
        "This is clearly supporting the claim. SUPPORT.",
        "Random gibberish with no label"
    ]
    for resp in test_responses:
        print(f"  '{resp[:50]}...' -> {parse_stance_response(resp)}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SDQC classification via prompting")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        choices=list(MODEL_OPTIONS.keys()),
                        help="Model to use")
    parser.add_argument("--test-on", type=str, default="test",
                        choices=["test", "dev"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--quick-test", action="store_true",
                        help="Run quick prompt test without loading model")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results to disk")
    
    args = parser.parse_args()
    
    if args.quick_test:
        quick_test()
    else:
        run_experiment(
            model_key=args.model,
            test_on=args.test_on,
            save=not args.no_save
        )
