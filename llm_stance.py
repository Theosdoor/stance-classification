import argparse
import torch
import json
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from data_loader import RumourEvalDataset
from tqdm import tqdm

from nb_main import LLM_NAME


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1) # LLM generation is usually sequential or small batch
    parser.add_argument('--limit', type=int, default=None, help="Limit number of samples for testing")
    args = parser.parse_args()

    # Load Test Data
    test_data_root = 'data/semeval2017-task8-test-data'
    test_labels_path = 'data/subtaska.json'
    print("Loading test dataset...")
    test_dataset = RumourEvalDataset(test_data_root, test_labels_path)
    print(f"Loaded {len(test_dataset)} test samples.")

    # Initialize Pipeline
    print(f"Loading model {LLM_NAME}...")
    pipe = pipeline("text-generation", model=LLM_NAME, torch_dtype=torch.float16, device_map="auto")

    predictions = []
    labels = []
    
    valid_labels = ['support', 'deny', 'query', 'comment']

    print("Running inference...")
    if args.limit:
        limit = min(args.limit, len(test_dataset))
        print(f"Limiting to first {limit} samples.")
        dataset_range = range(limit)
    else:
        dataset_range = range(len(test_dataset))

    for i in tqdm(dataset_range):
        sample = test_dataset[i]
        source_text = sample['source_text']
        reply_text = sample['reply_text']
        true_label = sample['label_text']
        
        # Construct Prompt
        messages = [
            {
                "role": "system",
                "content": "You are a stance detection classifier. Your task is to determine if a reply supports, denies, queries, or comments on a source tweet.",
            },
            {
                "role": "user",
                "content": f"Source Tweet: {source_text}\nReply Tweet: {reply_text}\n\nDoes the reply Support, Deny, Query, or Comment on the source? Output ONLY one of these four words.",
            },
        ]
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate
        outputs = pipe(prompt, max_new_tokens=10, do_sample=False, temperature=0.0) # Greedy decoding for deterministic output
        generated_text = outputs[0]["generated_text"]
        # Extract the assistant's response (everything after the prompt)
        # The pipeline output includes the prompt if not handled carefully, 
        # usually with chat models it returns the full text. 
        # We can split by the assistant token if needed, or just take the added text.
        # Transformers pipeline returns 'generated_text' which is the FULL conversation.
        
        response = generated_text[len(prompt):].strip().lower()
        
        # Simple parsing
        pred_label = 'comment' # Default fallback
        for v_label in valid_labels:
            if v_label in response:
                pred_label = v_label
                break
        
        predictions.append(pred_label)
        labels.append(true_label)

    # Evaluate
    print("\nEvaluating...")
    # Map back to IDs for sklearn or just use strings
    
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    
    # Confusion Matrix
    unique_labels = sorted(list(set(labels + predictions)))
    cm = confusion_matrix(labels, predictions, labels=unique_labels)
    print(f"\nConfusion Matrix (labels={unique_labels}):\n", cm)

if __name__ == "__main__":
    main()
