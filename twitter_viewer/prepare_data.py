#!/usr/bin/env python3
"""
Prepare SemEval tweet data for the static Twitter viewer.

Extracts all tweets from the dataset and exports them as JSON for the webpage.
"""

import os
import json
from pathlib import Path
from datetime import datetime

# Data paths
TRAIN_DATA_ROOT = '../data/semeval2017-task8-dataset/rumoureval-data'
TEST_DATA_ROOT = '../data/semeval2017-task8-test-data'
TRAIN_LABELS = '../data/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-train.json'
DEV_LABELS = '../data/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-dev.json'
TEST_LABELS = '../data/subtaska.json'

KNOWN_TOPICS = ['charliehebdo', 'ebola-essien', 'ferguson', 'germanwings-crash',
                'ottawashooting', 'prince-toronto', 'putinmissing', 'sydneysiege']


def load_json(path):
    """Load a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_tweet(path):
    """Load tweet data from a JSON file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def format_timestamp(created_at):
    """Convert Twitter timestamp to readable format."""
    try:
        # Twitter format: "Sun Aug 10 02:24:03 +0000 2014"
        dt = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
        return dt.strftime("%b %d, %Y · %I:%M %p")
    except:
        return created_at


def extract_tweet_info(tweet_data):
    """Extract relevant info from tweet JSON."""
    if not tweet_data:
        return None
    
    user = tweet_data.get('user', {})
    return {
        'id': str(tweet_data.get('id', '')),
        'text': tweet_data.get('text', ''),
        'created_at': format_timestamp(tweet_data.get('created_at', '')),
        'retweet_count': tweet_data.get('retweet_count', 0),
        'favorite_count': tweet_data.get('favorite_count', 0),
        'in_reply_to_status_id': str(tweet_data.get('in_reply_to_status_id', '')) if tweet_data.get('in_reply_to_status_id') else None,
        'user': {
            'name': user.get('name', 'Unknown'),
            'screen_name': user.get('screen_name', 'unknown'),
            'profile_image_url': user.get('profile_image_url_https', user.get('profile_image_url', '')),
            'verified': user.get('verified', False),
            'followers_count': user.get('followers_count', 0)
        }
    }


def load_context_data(thread_path):
    """Load context data (urls.dat and wikipedia) for a thread."""
    context = {}
    
    # Load urls.dat - contains MD5 hash -> URL mappings
    urls_dat_path = os.path.join(thread_path, 'urls.dat')
    if os.path.exists(urls_dat_path):
        urls = []
        try:
            with open(urls_dat_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        urls.append({
                            'hash': parts[0],
                            'short_url': parts[1] if len(parts) > 1 else '',
                            'expanded_url': parts[2] if len(parts) > 2 else parts[1]
                        })
            if urls:
                context['urls'] = urls
        except Exception as e:
            print(f"Error loading urls.dat from {thread_path}: {e}")
    
    # Load wikipedia context file
    wiki_path = os.path.join(thread_path, 'context', 'wikipedia')
    if os.path.exists(wiki_path):
        try:
            with open(wiki_path, 'r', encoding='utf-8') as f:
                wiki_content = f.read()
                cleaned = clean_wikitext(wiki_content)
                if cleaned and len(cleaned) > 50:  # Only include if meaningful content
                    context['wikipedia'] = cleaned[:600]  # Slightly longer limit for cleaned text
        except Exception as e:
            print(f"Error loading wikipedia from {thread_path}: {e}")
    
    return context if context else None


def clean_wikitext(text):
    """Clean WikiText markup to produce readable plain text."""
    import re
    
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # Remove templates like {{...}} (including nested)
    # Process multiple times for nested templates
    for _ in range(5):
        text = re.sub(r'\{\{[^{}]*\}\}', '', text)
    
    # Remove remaining unmatched {{ or }}
    text = re.sub(r'\{\{|\}\}', '', text)
    
    # Convert [[link|display]] to display
    text = re.sub(r'\[\[([^|\]]*\|)?([^\]]+)\]\]', r'\2', text)
    
    # Remove file/image links
    text = re.sub(r'\[\[(?:File|Image):[^\]]+\]\]', '', text, flags=re.IGNORECASE)
    
    # Remove external links [url text] -> text
    text = re.sub(r'\[https?://[^\s\]]+ ([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://[^\]]+\]', '', text)
    
    # Remove references <ref>...</ref>
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<ref[^/>]*/>', '', text, flags=re.IGNORECASE)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove bold/italic wiki markup
    text = re.sub(r"'''?", '', text)
    
    # Remove section headers
    text = re.sub(r'==+[^=]+=+', '', text)
    
    # Remove category links
    text = re.sub(r'\[\[Category:[^\]]+\]\]', '', text, flags=re.IGNORECASE)
    
    # Clean up whitespace
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'  +', ' ', text)
    text = text.strip()
    
    # Get first meaningful paragraph(s)
    lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 30]
    
    return ' '.join(lines[:3]) if lines else ''


def build_reply_tree(structure, replies_dir, labels):
    """Build nested reply structure from structure.json."""
    replies = []
    
    if isinstance(structure, dict):
        for reply_id, nested in structure.items():
            reply_path = os.path.join(replies_dir, f"{reply_id}.json")
            if os.path.exists(reply_path):
                tweet_data = load_tweet(reply_path)
                tweet_info = extract_tweet_info(tweet_data)
                if tweet_info:
                    tweet_info['stance'] = labels.get(reply_id, 'unknown')
                    tweet_info['replies'] = build_reply_tree(nested, replies_dir, labels)
                    replies.append(tweet_info)
    elif isinstance(structure, list):
        # Empty list means no more replies
        pass
    
    return replies


def process_thread(thread_path, labels):
    """Process a single thread directory."""
    source_dir = os.path.join(thread_path, 'source-tweet')
    replies_dir = os.path.join(thread_path, 'replies')
    structure_path = os.path.join(thread_path, 'structure.json')
    
    if not os.path.exists(source_dir):
        return None
    
    # Load source tweet
    source_files = [f for f in os.listdir(source_dir) if f.endswith('.json')]
    if not source_files:
        return None
    
    source_id = source_files[0].replace('.json', '')
    source_data = load_tweet(os.path.join(source_dir, source_files[0]))
    source_info = extract_tweet_info(source_data)
    
    if not source_info:
        return None
    
    source_info['stance'] = labels.get(source_id, 'source')
    
    # Load structure and build reply tree
    if os.path.exists(structure_path):
        structure = load_json(structure_path)
        # Structure has source_id as root key
        if source_id in structure:
            source_info['replies'] = build_reply_tree(structure[source_id], replies_dir, labels)
        else:
            source_info['replies'] = []
    else:
        source_info['replies'] = []
    
    # Load context data for this thread
    context = load_context_data(thread_path)
    if context:
        source_info['context'] = context
    
    return source_info


def process_topic(topic_path, labels):
    """Process all threads in a topic directory."""
    threads = []
    
    for item in os.listdir(topic_path):
        thread_path = os.path.join(topic_path, item)
        if os.path.isdir(thread_path):
            thread = process_thread(thread_path, labels)
            if thread:
                threads.append(thread)
    
    # Sort by timestamp (newest first)
    threads.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return threads


def main():
    """Main function to process all data and export JSON."""
    # Load all labels
    all_labels = {}
    for label_path in [TRAIN_LABELS, DEV_LABELS, TEST_LABELS]:
        if os.path.exists(label_path):
            all_labels.update(load_json(label_path))
    
    print(f"Loaded {len(all_labels)} labels")
    
    # Process training topics
    data = {'topics': {}}
    
    for topic in KNOWN_TOPICS:
        topic_path = os.path.join(TRAIN_DATA_ROOT, topic)
        if os.path.exists(topic_path):
            threads = process_topic(topic_path, all_labels)
            data['topics'][topic] = {
                'name': topic.replace('-', ' ').title(),
                'threads': threads,
                'thread_count': len(threads)
            }
            print(f"Processed {topic}: {len(threads)} threads")
    
    # Process test data
    test_threads = []
    for item in os.listdir(TEST_DATA_ROOT):
        thread_path = os.path.join(TEST_DATA_ROOT, item)
        if os.path.isdir(thread_path):
            thread = process_thread(thread_path, all_labels)
            if thread:
                test_threads.append(thread)
    
    test_threads.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    data['topics']['test'] = {
        'name': 'Test Data',
        'threads': test_threads,
        'thread_count': len(test_threads)
    }
    print(f"Processed test data: {len(test_threads)} threads")
    
    # Export to JSON
    output_path = 'data.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\nExported data to {output_path}")
    
    # Print summary
    total_threads = sum(t['thread_count'] for t in data['topics'].values())
    print(f"Total: {len(data['topics'])} topics, {total_threads} threads")


if __name__ == '__main__':
    main()
