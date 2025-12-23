# %%
import os
import re
import json
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import seaborn as sns
import matplotlib.pyplot as plt

from data_loader import (
    load_dataset, format_input_with_context,
    LABEL_TO_ID, ID_TO_LABEL
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# %%
# DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = torch.device(DEVICE_NAME)


# models ----
# meta-llama/Llama-3.1-8B-Instruct
# google/gemma-3-12b-it

MODEL_NAME = "google/gemma-3-12b-it"

pipe = pipeline("text-generation", MODEL_NAME, device_map='auto')

# %%
# prompts
# defns from paper https://www.derczynski.com/sheffield/papers/rumoureval/1704.07221.pdf
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

VALID_STANCES = {"SUPPORT", "DENY", "QUERY", "COMMENT"}

def parse_response(response_text):
    response_text = response_text.strip().upper()
    
    # Check for exact match first
    if response_text in VALID_STANCES:
        return response_text
    
    # Look for stance word in the response
    for stance in VALID_STANCES:
        if stance in response_text:
            return stance
    
    return None  # Parsing failed

# prompt template (used for both zero-shot and few-shot)
USER_PROMPT_TEMPLATE = """\
**Thread Context:**
{thread_context}

**Task:** Classify the stance of [TARGET] towards [SRC].
"""

def build_prompt(thread_context):
    """Build a user prompt for stance classification."""
    return USER_PROMPT_TEMPLATE.format(thread_context=thread_context)

# %%
# few shot examples - selection strategy:
# 1. Prefer examples with context (depth > 1, has parent)
# 2. From different topics for diversity
# 3. Unambiguous stance signals
train_df, dev_df, test_df = load_dataset()

def select_few_shot_examples(df):
    """Select diverse, high-quality few-shot examples.
    
    Strategy:
    - Prefer examples with thread context (depth > 1)
    - Select from different topics
    - One example per stance class
    """

    
    examples = {}
    used_topics = set()
    
    for label in ['support', 'deny', 'query', 'comment']:
        class_df = df[df['label_text'] == label].copy()
        
        # Prefer examples with context (depth > 1 means has parent)
        with_context = class_df[class_df['depth'] > 1]
        
        # Try to pick from different topics
        for pool in [with_context, class_df]:
            if len(pool) == 0:
                continue
            # Prefer unused topics
            unused = pool[~pool['topic'].isin(used_topics)]
            if len(unused) > 0:
                selected = unused.sample(1).iloc[0]
            else:
                selected = pool.sample(1).iloc[0]
            examples[label] = selected
            used_topics.add(selected['topic'])
            break
    
    return examples

FEW_SHOT_EXAMPLES = select_few_shot_examples(train_df)

def build_few_shot_messages(input_text):
    """Build messages with few-shot examples for stance classification."""
    messages = [
        {"role": "system", "content": SYS_PROMPT},
    ]
    
    # Add examples in order: support, deny, query, comment
    for label in ['support', 'deny', 'query', 'comment']:
        example = FEW_SHOT_EXAMPLES[label]
        example_input = format_input_with_context(example, train_df, use_features=False)
        messages.append({"role": "user", "content": build_prompt(example_input)})
        messages.append({"role": "assistant", "content": label.upper()})
    
    # Add actual query
    messages.append({"role": "user", "content": build_prompt(input_text)})
    
    return messages

def build_zero_shot_messages(input_text):
    """Build messages for zero-shot stance classification."""
    return [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": build_prompt(input_text, '')},
    ]

# %%
# Test sample
test_sample = test_df[test_df['label_text']=='deny'].iloc[0]
print(f"Test sample - True label: {test_sample['label_text'].upper()}")
print(f"Text: {test_sample['text'][:100]}...")
print()

# %%
# Zero-shot test
print("=" * 50)
print("ZERO-SHOT TEST")
print("=" * 50)

test_input = format_input_with_context(test_sample, test_df, use_features=False)
zero_shot_messages = build_zero_shot_messages(test_input)
output = pipe(zero_shot_messages)
raw_response = output[0]["generated_text"][-1]["content"].strip()
print("Raw response:", raw_response)

predicted = parse_response(raw_response)
print("Predicted:", predicted if predicted else "PARSE_ERROR")
print()

# %%
# Few-shot test
print("=" * 50)
print("FEW-SHOT TEST")
print("=" * 50)

few_shot_messages = build_few_shot_messages(test_input)
output = pipe(few_shot_messages)
raw_response = output[0]["generated_text"][-1]["content"].strip()
print("Raw response:", raw_response)

predicted = parse_response(raw_response)
print("Predicted:", predicted if predicted else "PARSE_ERROR")

# %%
