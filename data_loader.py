import os
import re
import json
import pickle
import urllib.request

import pandas as pd
import nltk
from emoji import demojize
from nltk.tokenize import TweetTokenizer

nltk.download('punkt', quiet=True)

# params

LABEL2ID = {'support': 0, 'deny': 1, 'query': 2, 'comment': 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

TOPICS = {'charliehebdo', 'ebola-essien', 'ferguson', 'germanwings-crash',
            'ottawashooting', 'prince-toronto', 'putinmissing', 'sydneysiege'}

# data paths
TRAIN_DATA_ROOT = 'downloaded_data/semeval2017-task8-dataset/rumoureval-data'
TRAIN_LABELS = 'downloaded_data/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-train.json'
DEV_LABELS = 'downloaded_data/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-dev.json'
TEST_DATA_ROOT = 'downloaded_data/semeval2017-task8-test-data'
TEST_LABELS = 'downloaded_data/subtaska.json'
SAVED_DATA_DIR = 'saved_data'

# for feature extraction (from https://github.com/kochkinaelena/branchLSTM/blob/master/preprocessing.py)
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

# load bad words (from branchlstm code)
BADWORDS = None
def _get_swear_words():
    global BADWORDS
    if BADWORDS is None:
        BADWORDS = set()
        try:
            with urllib.request.urlopen('https://raw.githubusercontent.com/kochkinaelena/branchLSTM/refs/heads/master/badwords.txt') as response:
                content = response.read().decode('utf-8')
                BADWORDS = {line.strip().lower() for line in content.splitlines() if line.strip()}
        except Exception as e:
            print(f"Warning: Could not fetch badwords from URL: {e}")
    return BADWORDS


# extract features 
def extract_features(tweet_json, depth=0):
    '''Extract features from a tweet JSON object.
    
    Adapted from https://github.com/kochkinaelena/branchLSTM/blob/master/preprocessing.py
    '''
    text = tweet_json.get('text', '') or ''
    user = tweet_json.get('user', {}) or {}
    
    tokens = set(re.sub(r'([^\s\w]|_)+', '', text.lower()).split())
    
    features = {
        'user_verified': user.get('verified', False),
        'user_has_url': bool(user.get('url')),
        'user_default_profile': user.get('default_profile', False),
        'user_followers_count': user.get('followers_count', 0),
        'user_friends_count': user.get('friends_count', 0),
        'user_listed_count': user.get('listed_count', 0),
        'user_statuses_count': user.get('statuses_count', 0),
        'user_has_description': bool(user.get('description')),
        
        'retweet_count': tweet_json.get('retweet_count', 0),
        'favorite_count': tweet_json.get('favorite_count', 0),
        'depth': depth,
        
        'has_qmark': '?' in text,
        'has_emark': '!' in text,
        'has_hashtag': '#' in text,
        'has_url': 'http' in text or 't.co' in text,
        'has_pic': 'pic.twitter.com' in text or 'instagr.am' in text,
        'has_RT': text.startswith('RT ') or text.startswith('MT '),
        'char_count': len(text),
        'word_count': len(text.split()),
        
        'has_negation': bool(tokens & NEGATION_WORDS),
        'num_negation': len(tokens & NEGATION_WORDS),
        'has_swearwords': bool(tokens & _get_swear_words()),
        'num_false_synonyms': len(tokens & FALSE_SYNONYMS),
        'num_false_antonyms': len(tokens & FALSE_ANTONYMS),
        'has_unconfirmed': 'unconfirmed' in tokens,
        'has_rumour': bool(tokens & {'rumour', 'rumor', 'gossip', 'hoax'}),
        'num_wh_words': len(tokens & WH_WORDS),
        'capital_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
    }
    
    return features


# process twitter thread

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
    
    source_files = [f for f in os.listdir(source_dir) if f.endswith('.json')]
    if not source_files:
        return []
    
    source_id = source_files[0].replace('.json', '')
    source_json = load_tweet(os.path.join(source_dir, source_files[0]))
    if not source_json:
        return []
    
    structure = {}
    if os.path.exists(structure_path):
        structure = load_json(structure_path)
        if source_id in structure:
            structure = structure[source_id]
    
    reply_jsons = {}
    if os.path.exists(replies_dir):
        for reply_file in os.listdir(replies_dir):
            if reply_file.endswith('.json'):
                reply_id = reply_file.replace('.json', '')
                reply_json = load_tweet(os.path.join(replies_dir, reply_file))
                if reply_json:
                    reply_jsons[reply_id] = reply_json
    
    id_to_text = {source_id: source_json.get('text', '')}
    for rid, rjson in reply_jsons.items():
        id_to_text[rid] = rjson.get('text', '')
    
    parent_map = {}
    depth_map = {source_id: 0}
    
    def traverse_structure(struct, parent_id, current_depth):
        if isinstance(struct, dict):
            for child_id, nested in struct.items():
                parent_map[child_id] = parent_id
                depth_map[child_id] = current_depth
                traverse_structure(nested, child_id, current_depth + 1)
    
    traverse_structure(structure, source_id, 1)
    
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
            
        if len(path_to_source) <= 2:
            return []
        
        intermediate_ids = path_to_source[2:]
        intermediate_ids = intermediate_ids[::-1]
        
        context_texts = []
        for tid in intermediate_ids:
            if tid in id_to_text:
                context_texts.append(id_to_text[tid])
        
        return context_texts

    
    results = []
    
    if source_id in labels:
        results.append({
            'tweet_id': source_id,
            'source_id': None,
            'parent_id': None,
            'text': source_json.get('text', ''),
            'topic': topic,
            'label_text': labels[source_id],
            'label': LABEL2ID[labels[source_id]],
            'context_chain': [],
            'depth': 0,
            'features': extract_features(source_json, depth=0),
        })
    
    for reply_id, reply_json in reply_jsons.items():
        if reply_id not in labels:
            continue
        
        parent_id = parent_map.get(reply_id)
        if not parent_id:
            in_reply = reply_json.get('in_reply_to_status_id_str') or reply_json.get('in_reply_to_status_id')
            if in_reply:
                parent_id = str(in_reply)
        
        actual_parent_id = None if parent_id == source_id else parent_id
        
        depth = depth_map.get(reply_id, 1)
        
        results.append({
            'tweet_id': reply_id,
            'source_id': source_id,
            'parent_id': actual_parent_id,
            'text': reply_json.get('text', ''),
            'topic': topic,
            'label_text': labels[reply_id],
            'label': LABEL2ID[labels[reply_id]],
            'context_chain': get_context_chain(reply_id),
            'depth': depth,
            'features': extract_features(reply_json, depth=depth),
        })
    
    return results


def process_data_root(data_root, labels, is_test=False):
    """Process all threads in a data root directory."""
    all_tweets = []
    
    if is_test:
        for item in os.listdir(data_root):
            thread_path = os.path.join(data_root, item)
            if os.path.isdir(thread_path) and not item.startswith('.'):
                tweets = build_thread_data(thread_path, labels, topic='test')
                all_tweets.extend(tweets)
    else:
        for topic in TOPICS:
            topic_path = os.path.join(data_root, topic)
            if not os.path.exists(topic_path):
                continue
            
            for item in os.listdir(topic_path):
                thread_path = os.path.join(topic_path, item)
                if os.path.isdir(thread_path) and not item.startswith('.'):
                    tweets = build_thread_data(thread_path, labels, topic=topic)
                    all_tweets.extend(tweets)
    
    return pd.DataFrame(all_tweets)


# loading fns
def _load_all_labels():
    train_labels = load_json(TRAIN_LABELS) if os.path.exists(TRAIN_LABELS) else {}
    dev_labels = load_json(DEV_LABELS) if os.path.exists(DEV_LABELS) else {}
    test_labels = load_json(TEST_LABELS) if os.path.exists(TEST_LABELS) else {}
    return train_labels, dev_labels, test_labels


def _build_datasets():
    train_labels, dev_labels, test_labels = _load_all_labels()

    train_df = process_data_root(TRAIN_DATA_ROOT, train_labels, is_test=False)
    dev_df = process_data_root(TRAIN_DATA_ROOT, dev_labels, is_test=False)
    test_df = process_data_root(TEST_DATA_ROOT, test_labels, is_test=True)    
    return train_df, dev_df, test_df


def load_dataset(force_rebuild=False):
    cache_path = os.path.join(SAVED_DATA_DIR, 'datasets.pkl')
    
    # return saved if possible
    if not force_rebuild and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    train_df, dev_df, test_df = _build_datasets()
    
    os.makedirs(SAVED_DATA_DIR, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump((train_df, dev_df, test_df), f)
    print(f"Saved dataset cache to {cache_path}")
    
    return train_df, dev_df, test_df


# format features for model input
def format_feature_string(features, keys=None):
    if keys is None:
        keys = ['user_verified', 'depth', 'has_qmark', 'has_url', 'has_negation']
    
    parts = []
    for k in keys:
        v = features.get(k, 0)
        if isinstance(v, bool):
            v = int(v)
        if v:
            parts.append(f"{k.split('_')[-1]}:{v}")
    
    return ','.join(parts) if parts else 'none'


# tweet normalisation (from https://github.com/VinAIResearch/BERTweet/blob/master/TweetNormalizer.py)

_tweet_tokenizer = TweetTokenizer()

def _normalise_token(token):
    '''Normalize a single token.
    
    From https://github.com/VinAIResearch/BERTweet/blob/master/TweetNormalizer.py
    '''
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "'":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalise_tweet(tweet):
    '''Normalize tweet text for BERTweet.
    
    From https://github.com/VinAIResearch/BERTweet/blob/master/TweetNormalizer.py
    '''
    if not tweet:
        return tweet
    
    tokens = _tweet_tokenizer.tokenize(tweet.replace("'", "'").replace("…", "..."))
    norm_tweet = " ".join([_normalise_token(token) for token in tokens])

    norm_tweet = (
        norm_tweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    norm_tweet = (
        norm_tweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    norm_tweet = (
        norm_tweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    return " ".join(norm_tweet.split())



def format_input_with_context(row, df, use_features=True, use_context=True, max_tokens=None, tokenizer=None):
    """
    Format input for classifier with context and features.
    
    Format: [SRC] source [SRC_F] features [CTX] context [PARENT] parent [PARENT_F] features [TARGET] target [TARGET_F] features
    
    If max_tokens and tokenizer are provided, truncates context first if needed.

    From https://aclanthology.org/S19-2191.pdf
    """
    target_text = normalise_tweet(row['text'])
    source_id = row['source_id']
    parent_id = row['parent_id']
    context_chain = [normalise_tweet(ctx) for ctx in row.get('context_chain', [])]
    features = row.get('features', {})
    
    core_parts = []
    if source_id is not None:
        source_row = df[df['tweet_id'] == source_id]
        if len(source_row) > 0:
            source_text = normalise_tweet(source_row.iloc[0]['text'])
            core_parts.append(f"[SRC] {source_text}")
            if use_features:
                core_parts.append(f"[SRC_F] {format_feature_string(source_row.iloc[0].get('features', {}))}")
    
    parent_parts = []
    if parent_id is not None:
        parent_row = df[df['tweet_id'] == parent_id]
        if len(parent_row) > 0:
            parent_text = normalise_tweet(parent_row.iloc[0]['text'])
            parent_parts.append(f"[PARENT] {parent_text}")
            if use_features:
                parent_parts.append(f"[PARENT_F] {format_feature_string(parent_row.iloc[0].get('features', {}))}")
    
    target_parts = [f"[TARGET] {target_text}"]
    if use_features:
        target_parts.append(f"[TARGET_F] {format_feature_string(features)}")
    
    context_parts = []
    if use_context and context_chain:
        available_context = list(context_chain)
        
        if max_tokens and tokenizer:
            core_text = ' '.join(core_parts + parent_parts + target_parts)
            core_tokens = len(tokenizer.encode(core_text, add_special_tokens=False))
            remaining_tokens = max_tokens - core_tokens - 10
            
            while available_context and remaining_tokens > 0:
                ctx_text = ' [SEP] '.join(available_context)
                ctx_tokens = len(tokenizer.encode(f"[CTX] {ctx_text}", add_special_tokens=False))
                if ctx_tokens <= remaining_tokens:
                    break
                available_context = available_context[1:]
        else:
            available_context = available_context[:3]
        
        if available_context:
            ctx_text = ' [SEP] '.join(available_context)
            context_parts.append(f"[CTX] {ctx_text}")
    
    all_parts = core_parts + context_parts + parent_parts + target_parts
    return ' '.join(all_parts)


# get specific datasets

def get_train_data():
    train_df, _, _ = load_dataset()
    return train_df

def get_dev_data():
    _, dev_df, _ = load_dataset()
    return dev_df

def get_test_data():
    _, _, test_df = load_dataset()
    return test_df

def get_all_data():
    train_df, dev_df, test_df = load_dataset()
    return pd.concat([train_df, dev_df, test_df], ignore_index=True)
