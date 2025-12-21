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

DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_NAME)

# %%
MODEL_NAME = "google/gemma-3-1b-it"
pipe = pipeline("text-generation", MODEL_NAME, device_map='auto')

MAIN_PROMPT = """\
A 12-month-old girl is brought in by her mother to the pediatrician for the
first time since her 6-month checkup. The mother states that her daughter
had been doing fine, but the parents are now concerned that their daughter
is still not able to stand up or speak. On exam, the patient has a temperature of 98.5°F (36.9°C), pulse is 96/min, respirations are 20/min, and blood
pressure is 100/80 mmHg. The child appears to have difficulty supporting
herself while sitting. The patient has no other abnormal physical findings.
She plays by herself and is making babbling noises but does not respond
to her own name. She appears to have some purposeless motions. A previous clinic note documents typical development at her 6-month visit and
mentioned that the patient was sitting unsupported at that time. Which of
the following is the most likely diagnosis?

A) Language disorder
B) Rett syndrome
C) Fragile X syndrome
D) Trisomy 21
"""

messages = [
     {"role": "system", "content": "You are a smart and intelligent medical expert."},
     {"role": "user", "content": MAIN_PROMPT},
]

output = pipe(messages)
output[0]["generated_text"][-1]["content"].strip()

# %%
