"""
Data loader for RumourEval stance classification.

Loads tweets with thread context, features, and parent/source relationships.
Features extraction inspired by extract_thread_features.py (SemEval baseline).

Schema:
- tweet_id: str - Tweet ID
- source_id: str/None - Source tweet ID (None if this IS the source)
- parent_id: str/None - Direct parent tweet ID (None if parent=source or is source)
- text: str - Tweet text
- topic: str - Topic name
- label: int - 0-3 (S/D/Q/C)
- label_text: str - support/deny/query/comment
- context_chain: list[str] - Tweet texts between source→parent (excludes source, parent, target)
- depth: int - Depth in thread (0=source)
- features: dict - Extracted features
"""

import os
import re
import json
import pickle
import pandas as pd
from pathlib import Path

# ============================================================================
# Constants
# ============================================================================

LABEL_TO_ID = {'support': 0, 'deny': 1, 'query': 2, 'comment': 3}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

KNOWN_TOPICS = {'charliehebdo', 'ebola-essien', 'ferguson', 'germanwings-crash',
                'ottawashooting', 'prince-toronto', 'putinmissing', 'sydneysiege'}

# Data paths
TRAIN_DATA_ROOT = 'data/semeval2017-task8-dataset/rumoureval-data'
TRAIN_LABELS = 'data/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-train.json'
DEV_LABELS = 'data/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-dev.json'
TEST_DATA_ROOT = 'data/semeval2017-task8-test-data'
TEST_LABELS = 'data/subtaska.json'
SAVED_DATA_DIR = 'saved_data'

# Lexicons for feature extraction (from extract_thread_features.py)
NEGATION_WORDS = {'not', 'no', 'nobody', 'nothing', 'none', 'never',
                  'neither', 'nor', 'nowhere', 'hardly', 'scarcely',
                  'barely', 'don', 'isn', 'wasn', 'shouldn', 'wouldn',
                  'couldn', 'doesn'}

FALSE_SYNONYMS = {'false', 'bogus', 'deceitful', 'dishonest', 'distorted',
                  'erroneous', 'fake', 'fanciful', 'faulty', 'fictitious',
                  'fraudulent', 'improper', 'inaccurate', 'incorrect',
                  'invalid', 'misleading', 'mistaken', 'phony', 'specious',
                  'spurious', 'unfounded', 'unreal', 'untrue', 'untruthful',
                  'lie', 'lying', 'reject', 'sham', 'fishy'}

FALSE_ANTONYMS = {'accurate', 'authentic', 'correct', 'fair', 'faithful',
                  'frank', 'genuine', 'honest', 'moral', 'open', 'proven',
                  'real', 'right', 'sincere', 'sound', 'true', 'trustworthy',
                  'truthful', 'valid', 'actual', 'factual', 'just', 'known',
                  'precise', 'reliable', 'straight', 'substantiated'}

WH_WORDS = {'what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how'}

# Load swear words
_SWEAR_WORDS = None
def _get_swear_words():
    global _SWEAR_WORDS
    if _SWEAR_WORDS is None:
        _SWEAR_WORDS = set()
        swear_path = 'data_preprocessing/data/badwords.txt'
        if os.path.exists(swear_path):
            with open(swear_path, 'r') as f:
                _SWEAR_WORDS = {line.strip().lower() for line in f if line.strip()}
    return _SWEAR_WORDS


# ============================================================================
# Feature Extraction
# ============================================================================

def extract_features(tweet_json, depth=0):
    """
    Extract features from a tweet JSON object.
    
    Features inspired by extract_thread_features.py (SemEval baseline).
    """
    text = tweet_json.get('text', '') or ''
    user = tweet_json.get('user', {}) or {}
    
    # Tokenize for lexical features
    tokens = set(re.sub(r'([^\s\w]|_)+', '', text.lower()).split())
    
    features = {
        # User features
        'user_verified': user.get('verified', False),
        'user_has_url': bool(user.get('url')),
        'user_default_profile': user.get('default_profile', False),
        'user_followers_count': user.get('followers_count', 0),
        'user_friends_count': user.get('friends_count', 0),
        'user_listed_count': user.get('listed_count', 0),
        'user_statuses_count': user.get('statuses_count', 0),
        'user_has_description': bool(user.get('description')),
        
        # Tweet metadata
        'retweet_count': tweet_json.get('retweet_count', 0),
        'favorite_count': tweet_json.get('favorite_count', 0),
        'depth': depth,
        
        # Text features
        'has_qmark': '?' in text,
        'has_emark': '!' in text,
        'has_hashtag': '#' in text,
        'has_url': 'http' in text or 't.co' in text,
        'has_pic': 'pic.twitter.com' in text or 'instagr.am' in text,
        'has_RT': text.startswith('RT ') or text.startswith('MT '),
        'char_count': len(text),
        'word_count': len(text.split()),
        
        # Lexical features
        'has_negation': bool(tokens & NEGATION_WORDS),
        'num_negation': len(tokens & NEGATION_WORDS),
        'has_swearwords': bool(tokens & _get_swear_words()),
        'num_false_synonyms': len(tokens & FALSE_SYNONYMS),
        'num_false_antonyms': len(tokens & FALSE_ANTONYMS),
        'has_unconfirmed': 'unconfirmed' in tokens,
        'has_rumour': bool(tokens & {'rumour', 'rumor', 'gossip', 'hoax'}),
        'num_wh_words': len(tokens & WH_WORDS),
        
        # Capital ratio
        'capital_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
    }
    
    return features


# ============================================================================
# Thread Processing
# ============================================================================

def load_json(path):
    """Load a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_tweet(path):
    """Load tweet JSON from file."""
    try:
        return load_json(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def build_thread_data(thread_path, labels, topic=None):
    """
    Build tweet data for all tweets in a thread.
    
    Returns list of tweet dicts with context chain and features.
    """
    source_dir = os.path.join(thread_path, 'source-tweet')
    replies_dir = os.path.join(thread_path, 'replies')
    structure_path = os.path.join(thread_path, 'structure.json')
    
    if not os.path.exists(source_dir):
        return []
    
    # Load source tweet
    source_files = [f for f in os.listdir(source_dir) if f.endswith('.json')]
    if not source_files:
        return []
    
    source_id = source_files[0].replace('.json', '')
    source_json = load_tweet(os.path.join(source_dir, source_files[0]))
    if not source_json:
        return []
    
    # Load structure
    structure = {}
    if os.path.exists(structure_path):
        structure = load_json(structure_path)
        # Handle structure with source_id as root key
        if source_id in structure:
            structure = structure[source_id]
    
    # Load all replies
    reply_jsons = {}  # reply_id -> tweet_json
    if os.path.exists(replies_dir):
        for reply_file in os.listdir(replies_dir):
            if reply_file.endswith('.json'):
                reply_id = reply_file.replace('.json', '')
                reply_json = load_tweet(os.path.join(replies_dir, reply_file))
                if reply_json:
                    reply_jsons[reply_id] = reply_json
    
    # Build id -> text map for context chain building
    id_to_text = {source_id: source_json.get('text', '')}
    for rid, rjson in reply_jsons.items():
        id_to_text[rid] = rjson.get('text', '')
    
    # Build parent map from structure (BFS)
    parent_map = {}  # child_id -> parent_id
    depth_map = {source_id: 0}
    
    def traverse_structure(struct, parent_id, current_depth):
        if isinstance(struct, dict):
            for child_id, nested in struct.items():
                parent_map[child_id] = parent_id
                depth_map[child_id] = current_depth
                traverse_structure(nested, child_id, current_depth + 1)
    
    traverse_structure(structure, source_id, 1)
    
    # Build context chain for each tweet
    def get_context_chain(tweet_id):
        """Get texts from source to parent (excluding source, parent, and target).
        
        For a chain: source → A → B → C → tweet
        - parent = C
        - context_chain = [A, B] (intermediate tweets between source and parent)
        """
        if tweet_id not in parent_map:
            return []
        
        # Build path from tweet up to source
        path_to_source = []
        current = tweet_id
        while current in parent_map:
            path_to_source.append(current)
            current = parent_map[current]
        # path_to_source = [tweet, parent, grandparent, ..., first_reply_to_source]
        # current is now source_id
        
        # path_to_source[0] = tweet (target)
        # path_to_source[1] = parent (if exists)
        # path_to_source[2:] = intermediate tweets (from closest-to-parent to closest-to-source)
        
        if len(path_to_source) <= 2:
            # No intermediate tweets (direct reply or reply to direct reply)
            return []
        
        # Get intermediate tweet IDs (exclude target and parent)
        intermediate_ids = path_to_source[2:]  # grandparent, great-grandparent, ... first_reply
        # Reverse to get chronological order (closest to source first)
        intermediate_ids = intermediate_ids[::-1]
        
        # Get texts
        context_texts = []
        for tid in intermediate_ids:
            if tid in id_to_text:
                context_texts.append(id_to_text[tid])
        
        return context_texts

    
    results = []
    
    # Add source tweet if labeled
    if source_id in labels:
        results.append({
            'tweet_id': source_id,
            'source_id': None,  # This IS the source
            'parent_id': None,
            'text': source_json.get('text', ''),
            'topic': topic,
            'label_text': labels[source_id],
            'label': LABEL_TO_ID[labels[source_id]],
            'context_chain': [],
            'depth': 0,
            'features': extract_features(source_json, depth=0),
        })
    
    # Add reply tweets if labeled
    for reply_id, reply_json in reply_jsons.items():
        if reply_id not in labels:
            continue
        
        # Get parent - from structure or in_reply_to_status_id
        parent_id = parent_map.get(reply_id)
        if not parent_id:
            in_reply = reply_json.get('in_reply_to_status_id_str') or reply_json.get('in_reply_to_status_id')
            if in_reply:
                parent_id = str(in_reply)
        
        # Determine if parent is the source
        actual_parent_id = None if parent_id == source_id else parent_id
        
        depth = depth_map.get(reply_id, 1)
        
        results.append({
            'tweet_id': reply_id,
            'source_id': source_id,
            'parent_id': actual_parent_id,
            'text': reply_json.get('text', ''),
            'topic': topic,
            'label_text': labels[reply_id],
            'label': LABEL_TO_ID[labels[reply_id]],
            'context_chain': get_context_chain(reply_id),
            'depth': depth,
            'features': extract_features(reply_json, depth=depth),
        })
    
    return results


def process_data_root(data_root, labels, is_test=False):
    """Process all threads in a data root directory."""
    all_tweets = []
    
    if is_test:
        # Test data has threads directly in root
        for item in os.listdir(data_root):
            thread_path = os.path.join(data_root, item)
            if os.path.isdir(thread_path) and not item.startswith('.'):
                tweets = build_thread_data(thread_path, labels, topic='test')
                all_tweets.extend(tweets)
    else:
        # Train/dev data organized by topic
        for topic in KNOWN_TOPICS:
            topic_path = os.path.join(data_root, topic)
            if not os.path.exists(topic_path):
                continue
            
            for item in os.listdir(topic_path):
                thread_path = os.path.join(topic_path, item)
                if os.path.isdir(thread_path) and not item.startswith('.'):
                    tweets = build_thread_data(thread_path, labels, topic=topic)
                    all_tweets.extend(tweets)
    
    return pd.DataFrame(all_tweets)


# ============================================================================
# Main Loading Functions
# ============================================================================

def _load_all_labels():
    """Load all label files."""
    train_labels = load_json(TRAIN_LABELS) if os.path.exists(TRAIN_LABELS) else {}
    dev_labels = load_json(DEV_LABELS) if os.path.exists(DEV_LABELS) else {}
    test_labels = load_json(TEST_LABELS) if os.path.exists(TEST_LABELS) else {}
    return train_labels, dev_labels, test_labels


def _build_datasets():
    """Build train, dev, test DataFrames from raw data."""
    train_labels, dev_labels, test_labels = _load_all_labels()
    
    print(f"Loading train data ({len(train_labels)} labels)...")
    train_df = process_data_root(TRAIN_DATA_ROOT, train_labels, is_test=False)
    
    print(f"Loading dev data ({len(dev_labels)} labels)...")
    dev_df = process_data_root(TRAIN_DATA_ROOT, dev_labels, is_test=False)
    
    print(f"Loading test data ({len(test_labels)} labels)...")
    test_df = process_data_root(TEST_DATA_ROOT, test_labels, is_test=True)
    
    print(f"Loaded: train={len(train_df)}, dev={len(dev_df)}, test={len(test_df)}")
    
    return train_df, dev_df, test_df


def load_dataset(force_rebuild=False):
    """
    Load train, dev, test DataFrames.
    
    Uses cached pickle files in saved_data/ if available.
    
    Returns:
        tuple: (train_df, dev_df, test_df)
    """
    cache_path = os.path.join(SAVED_DATA_DIR, 'datasets.pkl')
    
    # Try to load from cache
    if not force_rebuild and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    # Build from raw data
    train_df, dev_df, test_df = _build_datasets()
    
    # Save to cache
    os.makedirs(SAVED_DATA_DIR, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump((train_df, dev_df, test_df), f)
    print(f"Saved cache to {cache_path}")
    
    return train_df, dev_df, test_df


# ============================================================================
# Input Formatting for Classifier
# ============================================================================

def format_feature_string(features, keys=None):
    """Format features as a compact string for model input."""
    if keys is None:
        keys = ['user_verified', 'depth', 'has_qmark', 'has_url', 'has_negation']
    
    parts = []
    for k in keys:
        v = features.get(k, 0)
        if isinstance(v, bool):
            v = int(v)
        if v:  # Only include non-zero features
            parts.append(f"{k.split('_')[-1]}:{v}")
    
    return ','.join(parts) if parts else 'none'


def format_input_with_context(row, df, use_features=True, use_context=True, max_tokens=None, tokenizer=None):
    """
    Format input for classifier with context and features.
    
    Format: [SRC] source [SRC_F] features [CTX] context [PARENT] parent [PARENT_F] features [TARGET] target [TARGET_F] features
    
    If max_tokens and tokenizer are provided, truncates context first if needed.
    """
    target_text = row['text']
    source_id = row['source_id']
    parent_id = row['parent_id']
    context_chain = row.get('context_chain', [])
    features = row.get('features', {})
    
    # Build core parts (source, parent, target) - these are never truncated
    core_parts = []
    
    # Source text (look up in df if this is not the source)
    if source_id is not None:
        source_row = df[df['tweet_id'] == source_id]
        if len(source_row) > 0:
            source_text = source_row.iloc[0]['text']
            core_parts.append(f"[SRC] {source_text}")
            if use_features:
                core_parts.append(f"[SRC_F] {format_feature_string(source_row.iloc[0].get('features', {}))}")
    
    # Build parent part
    parent_parts = []
    if parent_id is not None:
        parent_row = df[df['tweet_id'] == parent_id]
        if len(parent_row) > 0:
            parent_text = parent_row.iloc[0]['text']
            parent_parts.append(f"[PARENT] {parent_text}")
            if use_features:
                parent_parts.append(f"[PARENT_F] {format_feature_string(parent_row.iloc[0].get('features', {}))}")
    
    # Build target part
    target_parts = [f"[TARGET] {target_text}"]
    if use_features:
        target_parts.append(f"[TARGET_F] {format_feature_string(features)}")
    
    # Build context part - this gets truncated if needed
    context_parts = []
    if use_context and context_chain:
        # Start with all context, truncate if needed
        available_context = list(context_chain)  # Copy
        
        if max_tokens and tokenizer:
            # Estimate tokens for core parts
            core_text = ' '.join(core_parts + parent_parts + target_parts)
            core_tokens = len(tokenizer.encode(core_text, add_special_tokens=False))
            remaining_tokens = max_tokens - core_tokens - 10  # Leave buffer
            
            # Progressively reduce context until it fits
            while available_context and remaining_tokens > 0:
                ctx_text = ' [SEP] '.join(available_context)
                ctx_tokens = len(tokenizer.encode(f"[CTX] {ctx_text}", add_special_tokens=False))
                
                if ctx_tokens <= remaining_tokens:
                    break
                # Remove oldest context first
                available_context = available_context[1:]
        else:
            # No tokenizer - just limit to 3 context tweets
            available_context = available_context[:3]
        
        if available_context:
            ctx_text = ' [SEP] '.join(available_context)
            context_parts.append(f"[CTX] {ctx_text}")
    
    # Combine all parts in order: source -> context -> parent -> target
    all_parts = core_parts + context_parts + parent_parts + target_parts
    return ' '.join(all_parts)


# ============================================================================
# Legacy Functions (for backward compatibility)
# ============================================================================

def get_train_data():
    """Load training data."""
    train_df, _, _ = load_dataset()
    return train_df

def get_dev_data():
    """Load dev data."""
    _, dev_df, _ = load_dataset()
    return dev_df

def get_test_data():
    """Load test data."""
    _, _, test_df = load_dataset()
    return test_df

def get_all_data():
    """Load all data combined."""
    train_df, dev_df, test_df = load_dataset()
    return pd.concat([train_df, dev_df, test_df], ignore_index=True)


if __name__ == '__main__':
    # Test the loader
    train_df, dev_df, test_df = load_dataset(force_rebuild=True)
    print(f"\nTrain: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
    print(f"Columns: {train_df.columns.tolist()}")
    print(f"\nSource tweets in train: {train_df['source_id'].isna().sum()}")
    print(f"\nSample row:")
    print(train_df.iloc[0])
