import os
import json
import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict, List, Tuple

def load_labels(path: str) -> Dict[str, str]:
    with open(path, 'r') as f:
        return json.load(f)

def read_tweet_text(path: str) -> str:
    with open(path, 'r') as f:
        data = json.load(f)
    return data['text']

class RumourEvalDataset(Dataset):
    def __init__(self, data_path: str, label_path: str, transform=None):
        """
        Args:
            data_path (str): Path to the 'rumoureval-data' directory.
            label_path (str): Path to the label JSON file (e.g., 'rumoureval-subtaskA-train.json').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_path = data_path
        self.labels = load_labels(label_path)
        self.samples: List[Tuple[str, str, str]] = [] # (source_text, reply_text, label)
        self.label_to_id = {'support': 0, 'deny': 1, 'query': 2, 'comment': 3}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
        self._load_data()

    def _load_data(self):
        # Walk through the directory to find all 'source-tweet' folders
        for root, dirs, files in os.walk(self.data_path):
            if 'source-tweet' in dirs and 'replies' in dirs:
                # This is a thread directory
                thread_path = root
                
                # Get source tweet text
                source_tweet_dir = os.path.join(thread_path, 'source-tweet')
                
                # Assuming there is only one source tweet file in the directory
                if not os.path.exists(source_tweet_dir):
                    continue
                    
                source_files = [f for f in os.listdir(source_tweet_dir) if f.endswith('.json')]
                if not source_files:
                    continue
                source_file = source_files[0]
                source_text = read_tweet_text(os.path.join(source_tweet_dir, source_file))
                
                # Get replies
                replies_dir = os.path.join(thread_path, 'replies')
                if not os.path.exists(replies_dir):
                    continue
                
                for reply_file in os.listdir(replies_dir):
                    if not reply_file.endswith('.json'):
                        continue
                    
                    reply_id = reply_file.replace('.json', '')
                    
                    # Check if we have a label for this reply
                    if reply_id in self.labels:
                        reply_path = os.path.join(replies_dir, reply_file)
                        reply_text = read_tweet_text(reply_path)
                        label = self.labels[reply_id]
                        
                        self.samples.append((source_text, reply_text, label))
                        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        source_text, reply_text, label = self.samples[idx]
        
        # You might want to tokenize here or in a collate_fn, 
        # but returning raw text allows for flexibility with different tokenizers
        return {
            'source_text': source_text,
            'reply_text': reply_text,
            'label_text': label,
            'label': self.label_to_id[label]
        }

if __name__ == '__main__':
    # Test loading
    print("Loading Training Data...")
    data_root = 'data/semeval2017-task8-dataset/rumoureval-data'
    train_labels = 'data/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-train.json'
    
    dataset = RumourEvalDataset(data_root, train_labels)
    print(f"Loaded {len(dataset)} training samples")
    if len(dataset) > 0:
        print("Sample 0:", dataset[0])

    print("\nLoading Test Data...")
    test_root = 'data/semeval2017-task8-test-data'
    test_labels = 'data/subtaska.json'
    test_dataset = RumourEvalDataset(test_root, test_labels)
    print(f"Loaded {len(test_dataset)} test samples")
    if len(test_dataset) > 0:
        print("Sample 0:", test_dataset[0])
