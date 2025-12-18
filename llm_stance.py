import argparse
import torch
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from data_loader import RumourEvalDataset
from tqdm import tqdm

# Hardcoded Few-Shot Examples (One per class)
FEW_SHOT_EXAMPLES = [
    {
        "source": "Breaking: Explosion reported at the central station.",
        "reply": "I heard it too, very loud bang!",
        "label": "support"
    },
    {
        "source": "Rumour: The prime minister has resigned.",
        "reply": "This is completely false, official sources deny it.",
        "label": "deny"
    },
    {
        "source": "New iPhone 20 release date leaked?",
        "reply": "Is there any actual proof of this source?",
        "label": "query"
    },
    {
        "source": "Snow storm approaching the coast.",
        "reply": "I hope schools are closed tomorrow.",
        "label": "comment"
    }
]

def format_few_shot_prompt(source, reply):
    prompt_text = "Here are some examples of stance detection:\n\n"
    for ex in FEW_SHOT_EXAMPLES:
        prompt_text += f"Source: {ex['source']}\nReply: {ex['reply']}\nLabel: {ex['label']}\n\n"
    
    prompt_text += f"Source: {source}\nReply: {reply}\nLabel:"
    return prompt_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--limit', type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument('--mode', type=str, default='zero-shot', choices=['zero-shot', 'few-shot', 'cot'], help="Prompting mode")
    args = parser.parse_args()

    # Load Test Data
    test_data_root = 'data/semeval2017-task8-test-data'
    test_labels_path = 'data/subtaska.json'
    print(f"Loading test dataset (Mode: {args.mode})...")
    test_dataset = RumourEvalDataset(test_data_root, test_labels_path)
    print(f"Loaded {len(test_dataset)} test samples.")

    # Initialize Pipeline
    print(f"Loading model {args.model_id}...")
    pipe = pipeline("text-generation", model=args.model_id, torch_dtype=torch.float16, device_map="auto")

    predictions = []
    labels = []
    
    print("Running inference...")
    range_iter = range(len(test_dataset))
    if args.limit:
        limit = min(args.limit, len(test_dataset))
        print(f"Limiting to first {limit} samples.")
        range_iter = range(limit)

    for i in tqdm(range_iter):
        sample = test_dataset[i]
        source_text = sample['source_text']
        reply_text = sample['reply_text']
        true_label = sample['label_text']
        
        # --- Prompt Construction Strategy ---
        messages = []
        
        if args.mode == 'zero-shot':
            messages = [
                {"role": "system", "content": "You are a stance detection classifier. Determine if a reply supports, denies, queries, or comments on a source tweet."},
                {"role": "user", "content": f"Source: {source_text}\nReply: {reply_text}\n\nStance (support/deny/query/comment):"}
            ]
            
        elif args.mode == 'few-shot':
            # Construct a few-shot message
            # We can put examples in the user prompt or system prompt.
            # Putting them in the user prompt as a single block is common.
            fs_content = format_few_shot_prompt(source_text, reply_text)
            messages = [
                 {"role": "system", "content": "You are a stance detection classifier. Output single word label."},
                 {"role": "user", "content": fs_content}
            ]
            
        elif args.mode == 'cot':
            # Two-stage CoT (simulated in one prompt for simplicity, or actual 2 calls)
            # The prompt asks for step-by-step reasoning.
             messages = [
                {"role": "system", "content": "You are a logical stance detector."},
                {"role": "user", "content": f"Source: {source_text}\nReply: {reply_text}\n\nTask: Step 1. Is this a Comment or a Stance (Support/Deny/Query)?\nStep 2. If Stance, which one?\n\nOutput logic then final label (support/deny/query/comment)."}
            ]

        # Apply Template
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate
        outputs = pipe(prompt, max_new_tokens=50 if args.mode == 'cot' else 10, do_sample=False)
        generated_text = outputs[0]["generated_text"]
        response = generated_text[len(prompt):].strip().lower()
        
        # --- Parsing Strategy ---
        pred_label = 'comment' # Default
        valid_labels = ['support', 'deny', 'query', 'comment']
        
        # Simple keyword search in response
        found_labels = [l for l in valid_labels if l in response]
        
        if args.mode == 'cot':
            # For CoT, we expect the reasoning and THEN the label.
            # We take the *last* valid label found, assuming it's the conclusion.
            if found_labels:
                pred_label = found_labels[-1]
        else:
            # For zero/few-shot, we hope the output IS the label or starts with it.
            if found_labels:
                pred_label = found_labels[0]
                
        predictions.append(pred_label)
        labels.append(true_label)

    # Evaluate
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    
    print(f"\nResults ID: {args.mode}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    
    unique_labels = sorted(list(set(labels + predictions)))
    cm = confusion_matrix(labels, predictions, labels=unique_labels)
    print(f"\nConfusion Matrix (labels={unique_labels}):\n", cm)

if __name__ == "__main__":
    main()
