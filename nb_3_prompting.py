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
    get_train_data, get_dev_data, get_test_data,
    LABEL_TO_ID, ID_TO_LABEL
)

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
Your task is to classify the stance of a reply tweet towards a rumour (mentioned in the source tweet).
The stance falls into exactly ONE of these four categories:

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
**Source Tweet:**
{source_text}

**Reply Tweet:**
{reply_text}
"""

def build_prompt(source_text, reply_text):
    """Build a user prompt for stance classification."""
    return USER_PROMPT_TEMPLATE.format(
        source_text=source_text,
        reply_text=reply_text
    )

# %%
# few shot examples
train_df = get_train_data()
support_example = train_df[train_df['label_text']=='support'].iloc[0]
deny_example = train_df[train_df['label_text']=='deny'].iloc[0]
query_example = train_df[train_df['label_text']=='query'].iloc[0]
comment_example = train_df[train_df['label_text']=='comment'].iloc[0]

def build_few_shot_messages(source_text, reply_text):
    """Build messages with few-shot examples for stance classification."""
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        # Example 1: Support
        {"role": "user", "content": build_prompt(support_example['source_text'], support_example['reply_text'])},
        {"role": "assistant", "content": "SUPPORT"},
        # Example 2: Deny
        {"role": "user", "content": build_prompt(deny_example['source_text'], deny_example['reply_text'])},
        {"role": "assistant", "content": "DENY"},
        # Example 3: Query
        {"role": "user", "content": build_prompt(query_example['source_text'], query_example['reply_text'])},
        {"role": "assistant", "content": "QUERY"},
        # Example 4: Comment
        {"role": "user", "content": build_prompt(comment_example['source_text'], comment_example['reply_text'])},
        {"role": "assistant", "content": "COMMENT"},
        # Actual query
        {"role": "user", "content": build_prompt(source_text, reply_text)},
    ]
    return messages

def build_zero_shot_messages(source_text, reply_text):
    """Build messages for zero-shot stance classification."""
    return [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": build_prompt(source_text, reply_text)},
    ]

# %%
# Test sample
test_df = get_test_data()
test_sample = test_df[test_df['label_text']=='deny'].iloc[0]
print(f"Test sample - True label: {test_sample['label_text'].upper()}")
print(f"Source: {test_sample['source_text']}...")
print(f"Reply: {test_sample['reply_text']}...")
print()

# %%
# Zero-shot test
print("=" * 50)
print("ZERO-SHOT TEST")
print("=" * 50)

zero_shot_messages = build_zero_shot_messages(
    test_sample['source_text'], 
    test_sample['reply_text']
)
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

few_shot_messages = build_few_shot_messages(
    test_sample['source_text'], 
    test_sample['reply_text']
)
output = pipe(few_shot_messages)
raw_response = output[0]["generated_text"][-1]["content"].strip()
print("Raw response:", raw_response)

predicted = parse_response(raw_response)
print("Predicted:", predicted if predicted else "PARSE_ERROR")

# %%
