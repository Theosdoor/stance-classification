"""
NB1: Analytics and Insights for RumourEval Task
Part (a): Unigram/bigram analysis and token distribution comparison
"""

import re
from collections import Counter
from data_loader import get_train_data, get_dev_data

# =============================================================================
# Text Preprocessing
# =============================================================================

def tokenize(text):
    """Simple tokenizer: lowercase, remove URLs, mentions, and punctuation."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split()
    return tokens

def get_bigrams(tokens):
    """Generate bigrams from a list of tokens."""
    return [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]

# =============================================================================
# Part (a): Unigrams and Bigrams by Stance Label
# =============================================================================

def compute_ngrams_by_stance(dataset):
    """
    Compute unigrams and bigrams for replies, grouped by stance label.
    
    Returns:
        unigrams: dict mapping label -> Counter of unigrams
        bigrams: dict mapping label -> Counter of bigrams
    """
    unigrams = {'support': Counter(), 'deny': Counter(), 'query': Counter(), 'comment': Counter()}
    bigrams = {'support': Counter(), 'deny': Counter(), 'query': Counter(), 'comment': Counter()}
    
    for i in range(len(dataset)):
        sample = dataset[i]
        label = sample['label_text']
        reply_tokens = tokenize(sample['reply_text'])
        
        unigrams[label].update(reply_tokens)
        bigrams[label].update(get_bigrams(reply_tokens))
    
    return unigrams, bigrams

def print_top_ngrams(unigrams, bigrams, n=15):
    """Print top unigrams and bigrams for each stance."""
    labels = ['support', 'deny', 'query', 'comment']
    
    print("=" * 80)
    print("TOP UNIGRAMS BY STANCE")
    print("=" * 80)
    for label in labels:
        print(f"\n{label.upper()} (top {n}):")
        for word, count in unigrams[label].most_common(n):
            print(f"  {word}: {count}")
    
    print("\n" + "=" * 80)
    print("TOP BIGRAMS BY STANCE")
    print("=" * 80)
    for label in labels:
        print(f"\n{label.upper()} (top {n}):")
        for bigram, count in bigrams[label].most_common(n):
            print(f"  {' '.join(bigram)}: {count}")

# =============================================================================
# Part (a): Token Distribution Comparison (Stance vs Non-Stance)
# =============================================================================

def compare_stance_vs_comment(dataset):
    """
    Compare token distributions between:
    - Stance classes (Support, Deny, Query) 
    - Non-stance class (Comment)
    
    For both source text and reply text.
    """
    stance_source = Counter()
    stance_reply = Counter()
    comment_source = Counter()
    comment_reply = Counter()
    
    stance_count = 0
    comment_count = 0
    
    for i in range(len(dataset)):
        sample = dataset[i]
        label = sample['label_text']
        source_tokens = tokenize(sample['source_text'])
        reply_tokens = tokenize(sample['reply_text'])
        
        if label in ['support', 'deny', 'query']:
            stance_source.update(source_tokens)
            stance_reply.update(reply_tokens)
            stance_count += 1
        else:  # comment
            comment_source.update(source_tokens)
            comment_reply.update(reply_tokens)
            comment_count += 1
    
    return {
        'stance': {'source': stance_source, 'reply': stance_reply, 'count': stance_count},
        'comment': {'source': comment_source, 'reply': comment_reply, 'count': comment_count}
    }

def print_distribution_comparison(distributions, n=20):
    """Print comparison of token distributions."""
    stance = distributions['stance']
    comment = distributions['comment']
    
    print("\n" + "=" * 80)
    print("TOKEN DISTRIBUTION: STANCE (S/D/Q) vs NON-STANCE (Comment)")
    print("=" * 80)
    print(f"\nSample counts: Stance={stance['count']}, Comment={comment['count']}")
    
    # Compare reply distributions
    print(f"\n{'='*40}")
    print("REPLY TEXT COMPARISON")
    print(f"{'='*40}")
    
    print(f"\nTop {n} tokens in STANCE replies:")
    for token, count in stance['reply'].most_common(n):
        print(f"  {token}: {count}")
    
    print(f"\nTop {n} tokens in COMMENT replies:")
    for token, count in comment['reply'].most_common(n):
        print(f"  {token}: {count}")
    
    # Find distinctive tokens (high in one, low in other)
    print(f"\n{'='*40}")
    print("DISTINCTIVE TOKENS")
    print(f"{'='*40}")
    
    # Normalize by count for fair comparison
    stance_total = sum(stance['reply'].values())
    comment_total = sum(comment['reply'].values())
    
    # Tokens more common in stance replies
    stance_distinctive = []
    for token, count in stance['reply'].most_common(100):
        stance_freq = count / stance_total
        comment_freq = comment['reply'].get(token, 0) / comment_total
        if stance_freq > 0 and comment_freq > 0:
            ratio = stance_freq / comment_freq
            if ratio > 1.5:
                stance_distinctive.append((token, ratio, count))
    
    print(f"\nTokens MORE common in stance replies (ratio > 1.5x):")
    for token, ratio, count in sorted(stance_distinctive, key=lambda x: -x[1])[:15]:
        print(f"  {token}: {ratio:.2f}x more frequent (count={count})")
    
    # Tokens more common in comment replies
    comment_distinctive = []
    for token, count in comment['reply'].most_common(100):
        comment_freq = count / comment_total
        stance_freq = stance['reply'].get(token, 0) / stance_total
        if comment_freq > 0 and stance_freq > 0:
            ratio = comment_freq / stance_freq
            if ratio > 1.5:
                comment_distinctive.append((token, ratio, count))
    
    print(f"\nTokens MORE common in comment replies (ratio > 1.5x):")
    for token, ratio, count in sorted(comment_distinctive, key=lambda x: -x[1])[:15]:
        print(f"  {token}: {ratio:.2f}x more frequent (count={count})")

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("Loading training data...")
    train_data = get_train_data()
    print(f"Loaded {len(train_data)} samples\n")
    
    # Part (a): Unigrams and bigrams by stance
    print("Computing unigrams and bigrams by stance...")
    unigrams, bigrams = compute_ngrams_by_stance(train_data)
    print_top_ngrams(unigrams, bigrams, n=15)
    
    # Part (a): Token distribution comparison
    print("\nComparing stance vs non-stance token distributions...")
    distributions = compare_stance_vs_comment(train_data)
    print_distribution_comparison(distributions, n=20)
