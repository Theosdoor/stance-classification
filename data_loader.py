import os
import json
import pandas as pd
from pathlib import Path

# =============================================================================
# Helper Functions
# =============================================================================

def load_labels(path):
    with open(path, 'r') as f:
        return json.load(f)

def read_tweet_text(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data['text']

def parse_urls_dat(path):
    """Parse a urls.dat file. Returns list of dicts with 'md5', 'short_url', 'long_url'."""
    urls = []
    if not os.path.exists(path):
        return urls
    
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                urls.append({'md5': parts[0], 'short_url': parts[1], 'long_url': parts[2]})
            elif len(parts) == 2:
                urls.append({'md5': parts[0], 'short_url': parts[1], 'long_url': parts[1]})
    return urls

def load_context_content(thread_path, url_info):
    """Load archived context content for URLs in a thread."""
    context = {}
    context_urls_dir = os.path.join(thread_path, 'context', 'urls')
    
    if os.path.exists(context_urls_dir):
        for url_entry in url_info:
            md5 = url_entry['md5']
            for ext in ['', '.html', '.txt', '.htm']:
                context_file = os.path.join(context_urls_dir, md5 + ext)
                if os.path.exists(context_file):
                    try:
                        with open(context_file, 'r', encoding='utf-8', errors='ignore') as f:
                            context[md5] = {'content': f.read(), 'url': url_entry['long_url']}
                        break
                    except Exception:
                        pass
    
    # Wikipedia context
    wiki_path = os.path.join(thread_path, 'context', 'wikipedia')
    if os.path.exists(wiki_path):
        if os.path.isfile(wiki_path):
            try:
                with open(wiki_path, 'r', encoding='utf-8', errors='ignore') as f:
                    context['wikipedia'] = {'content': f.read(), 'url': 'wikipedia'}
            except Exception:
                pass
        elif os.path.isdir(wiki_path):
            for wiki_file in os.listdir(wiki_path):
                wiki_file_path = os.path.join(wiki_path, wiki_file)
                if os.path.isfile(wiki_file_path):
                    try:
                        with open(wiki_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            context['wikipedia_' + wiki_file] = {'content': f.read(), 'url': 'wikipedia'}
                    except Exception:
                        pass
    return context

# =============================================================================
# Thread Cache
# =============================================================================

_thread_cache = {}

KNOWN_EVENTS = {'charliehebdo', 'ebola-essien', 'ferguson', 'germanwings-crash',
                'ottawashooting', 'prince-toronto', 'putinmissing', 'sydneysiege'}

def _get_thread_data(thread_path, load_context=False):
    """Get cached thread data or load it."""
    cache_key = (thread_path, load_context)
    if cache_key in _thread_cache:
        return _thread_cache[cache_key]
    
    source_tweet_dir = os.path.join(thread_path, 'source-tweet')
    if not os.path.exists(source_tweet_dir):
        return None
    
    source_files = [f for f in os.listdir(source_tweet_dir) if f.endswith('.json')]
    if not source_files:
        return None
    
    source_file = source_files[0]
    source_id = source_file.replace('.json', '')
    source_text = read_tweet_text(os.path.join(source_tweet_dir, source_file))
    
    # Extract event name from path
    event = None
    for part in Path(thread_path).parts:
        if part in KNOWN_EVENTS:
            event = part
            break
    
    # Load URL info and context
    urls_dat_path = os.path.join(thread_path, 'urls.dat')
    url_info = parse_urls_dat(urls_dat_path)
    context = load_context_content(thread_path, url_info) if load_context else {}
    
    thread_data = {
        'source_id': source_id,
        'source_text': source_text,
        'event': event,
        'urls': url_info,
        'context': context
    }
    
    _thread_cache[cache_key] = thread_data
    return thread_data

def clear_cache():
    """Clear the thread data cache."""
    global _thread_cache
    _thread_cache = {}

# =============================================================================
# Label Mapping
# =============================================================================

LABEL_TO_ID = {'support': 0, 'deny': 1, 'query': 2, 'comment': 3}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

# =============================================================================
# Data Loading Functions
# =============================================================================

def load_data(data_path, label_path, load_context=False):
    """
    Load RumourEval data as a pandas DataFrame.
    
    Columns: reply_id, event, source_text, source_id, reply_text, label_text, label, urls, context
    """
    labels = load_labels(label_path)
    
    # Build reply_id -> (thread_path, reply_path) map
    reply_map = {}
    for root, dirs, files in os.walk(data_path):
        if 'source-tweet' in dirs and 'replies' in dirs:
            thread_path = root
            replies_dir = os.path.join(thread_path, 'replies')
            if not os.path.exists(replies_dir):
                continue
            for reply_file in os.listdir(replies_dir):
                if reply_file.endswith('.json'):
                    reply_id = reply_file.replace('.json', '')
                    reply_map[reply_id] = (thread_path, os.path.join(replies_dir, reply_file))
    
    # Build rows for DataFrame
    rows = []
    for reply_id, label_text in labels.items():
        if reply_id not in reply_map:
            continue
        
        thread_path, reply_path = reply_map[reply_id]
        thread_data = _get_thread_data(thread_path, load_context)
        if thread_data is None:
            continue
        
        reply_text = read_tweet_text(reply_path)
        
        rows.append({
            'reply_id': reply_id,
            'event': thread_data['event'],
            'source_text': thread_data['source_text'],
            'source_id': thread_data['source_id'],
            'reply_text': reply_text,
            'label_text': label_text,
            'label': LABEL_TO_ID[label_text],
            'urls': thread_data['urls'],
            'context': thread_data['context']
        })
    
    return pd.DataFrame(rows)

# =============================================================================
# Data Paths
# =============================================================================

TRAIN_DATA_ROOT = 'data/semeval2017-task8-dataset/rumoureval-data'
TRAIN_LABELS = 'data/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-train.json'
DEV_LABELS = 'data/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-dev.json'
TEST_DATA_ROOT = 'data/semeval2017-task8-test-data'
TEST_LABELS = 'data/subtaska.json'

def get_train_data(load_context=False):
    """Load the training dataset as a DataFrame."""
    return load_data(TRAIN_DATA_ROOT, TRAIN_LABELS, load_context=load_context)

def get_dev_data(load_context=False):
    """Load the development dataset as a DataFrame."""
    return load_data(TRAIN_DATA_ROOT, DEV_LABELS, load_context=load_context)

def get_test_data(load_context=False):
    """Load the test dataset as a DataFrame."""
    return load_data(TEST_DATA_ROOT, TEST_LABELS, load_context=load_context)

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("Loading Training Data...")
    train_df = get_train_data()
    print(f"Loaded {len(train_df)} training samples")
    print(f"\nColumns: {list(train_df.columns)}")
    print(f"\nFirst row:\n{train_df.iloc[0]}")
    
    print("\n" + "="*50)
    print("Loading Dev Data...")
    dev_df = get_dev_data()
    print(f"Loaded {len(dev_df)} dev samples")
    
    print("\n" + "="*50)
    print("Loading Test Data...")
    test_df = get_test_data()
    print(f"Loaded {len(test_df)} test samples")
    
    print("\n" + "="*50)
    print("Label distribution (train):")
    print(train_df['label_text'].value_counts())

    all_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)

