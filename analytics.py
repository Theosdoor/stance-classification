# %%
import html
import re
import string
from collections import Counter # like dict except returns 0 rather than key error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk

from data_loader import get_train_data, get_dev_data, get_test_data, get_all_data

# nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')

# global var
RAND_SEED = 42
STOPWORDS = set(stopwords.words('english'))
PROTECTED_WORDS = {'isis', 'news', 'texas', 'paris'}  # words not to lemmatize
SAVE_DIR = './results/analytics/'
WNL = WordNetLemmatizer()

# %%
# functions

# =============================================================================
# Text Preprocessing
# =============================================================================

# lemmatization
def get_wordnet_pos(penn_tag):
    """penn treebank tag to wordnet part-of-speech tag (wnl only used wordnet)"""
    if penn_tag.startswith('J'):
        return wordnet.ADJ
    elif penn_tag.startswith('V'):
        return wordnet.VERB
    elif penn_tag.startswith('N'):
        return wordnet.NOUN
    elif penn_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN # assume noun

def lemmatize_tokens(tokens):
    """
    lemmatize tokens with part-of-speech tagging 
    (to differentiate betwen eg. verb claim and noun claim)
    """
    pos_tagged = nltk.pos_tag(tokens)
    return [
        word if word in PROTECTED_WORDS else WNL.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in pos_tagged
    ]

# tokenization
def tokenize(text, lemmatize=True):
    """Tokenize: lowercase, decode HTML entities, remove URLs, mentions, punctuation, stopwords, and optionally lemmatize."""
    text = text.lower()
    
    # decode HTML entities (&amp, etc.)
    text = html.unescape(text)
    
    # use regex to remove patterns
    re_patterns = [
        re.compile(r'https?://\S+|www\.\S+'), # url
        re.compile(r'@\w+'), # mentions
        re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0" # <-- these are unicode ranges
                u"\U000024C2-\U0001F251"
                "]+", flags=re.UNICODE),
    ]
    for pattern in re_patterns:
        text = pattern.sub(r'', text)
    
    # remove punct
    punct_table = str.maketrans('', '', string.punctuation)
    text = text.translate(punct_table)
    
    # tokenise & ignore stopwords
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    
    # lemmatize tokens
    if lemmatize:
        tokens = lemmatize_tokens(tokens)
    
    return tokens

def preprocess_for_lda(text):
    """Preprocess text for LDA (return as string for CountVectorizer)."""
    return ' '.join(tokenize(text))

def get_bigrams(tokens):
    """Generate bigrams from a list of tokens."""
    return [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]

# =============================================================================
# Part (a): Unigrams and Bigrams by Stance Label
# =============================================================================

def get_ngrams_by_stance(df):
    """Compute unigrams and bigrams for replies, grouped by stance label."""
    unigrams = {'support': Counter(), 'deny': Counter(), 'query': Counter(), 'comment': Counter()}
    bigrams = {'support': Counter(), 'deny': Counter(), 'query': Counter(), 'comment': Counter()}
    
    for _, row in df.iterrows():
        label = row['label_text']
        tokens = tokenize(row['reply_text'])
        unigrams[label].update(tokens)
        bigrams[label].update(get_bigrams(tokens))
    
    return unigrams, bigrams

def plot_ngrams_by_stance(unigrams, bigrams, top_n=10, save_dir=SAVE_DIR):
    """Plot bar charts for top unigrams and bigrams by stance."""
    labels = ['support', 'deny', 'query', 'comment']
    
    # Plot unigrams
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for ax, label in zip(axes, labels):
        top_words = unigrams[label].most_common(top_n)
        if top_words:
            words, counts = zip(*top_words)
            sns.barplot(x=list(counts), y=list(words), ax=ax, 
                       hue=list(words), legend=False)
            ax.set_xlabel('Count')
            ax.set_title(f'{label.upper()}')
        
    plt.suptitle('Top Unigrams by Stance', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_dir + 'unigrams_by_stance.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir}unigrams_by_stance.png")
    plt.show()
    
    # Plot bigrams
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for ax, label in zip(axes, labels):
        top_bigrams = bigrams[label].most_common(top_n)
        if top_bigrams:
            bigram_strs = [' '.join(b) for b, _ in top_bigrams]
            counts = [c for _, c in top_bigrams]
            sns.barplot(x=counts, y=bigram_strs, ax=ax,
                       hue=bigram_strs, legend=False)
            ax.set_xlabel('Count')
            ax.set_title(f'{label.upper()}')
    
    plt.suptitle('Top Bigrams by Stance', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_dir + 'bigrams_by_stance.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir}bigrams_by_stance.png")
    plt.show()

# =============================================================================
# Part (a): Token Distribution Comparison (Stance vs Non-Stance)
# =============================================================================

def compare_stance_vs_comment(df):
    """Compare token distributions: Stance (S/D/Q) vs Non-stance (Comment)."""
    stance_reply = Counter()
    comment_reply = Counter()
    stance_count = 0
    comment_count = 0
    
    for _, row in df.iterrows():
        label = row['label_text']
        tokens = tokenize(row['reply_text'])
        
        if label in ['support', 'deny', 'query']:
            stance_reply.update(tokens)
            stance_count += 1
        else:
            comment_reply.update(tokens)
            comment_count += 1
    
    return {
        'stance': {'reply': stance_reply, 'count': stance_count},
        'comment': {'reply': comment_reply, 'count': comment_count}
    }

def print_distribution_comparison(distributions, n=15):
    """Print comparison of token distributions."""
    stance = distributions['stance']
    comment = distributions['comment']
    
    print("\n" + "=" * 80)
    print("TOKEN DISTRIBUTION: STANCE (S/D/Q) vs NON-STANCE (Comment)")
    print("=" * 80)
    print(f"Sample counts: Stance={stance['count']}, Comment={comment['count']}")
    
    # Normalize for fair comparison
    stance_total = sum(stance['reply'].values())
    comment_total = sum(comment['reply'].values())
    
    # Find distinctive tokens
    print(f"\nTokens MORE common in STANCE replies:")
    stance_distinctive = []
    for token, count in stance['reply'].most_common(100):
        stance_freq = count / stance_total
        comment_freq = comment['reply'].get(token, 1) / comment_total
        ratio = stance_freq / comment_freq
        if ratio > 1.5:
            stance_distinctive.append((token, ratio, count))
    for token, ratio, count in sorted(stance_distinctive, key=lambda x: -x[1])[:n]:
        print(f"  {token}: {ratio:.2f}x (count={count})")
    
    print(f"\nTokens MORE common in COMMENT replies:")
    comment_distinctive = []
    for token, count in comment['reply'].most_common(100):
        comment_freq = count / comment_total
        stance_freq = stance['reply'].get(token, 1) / stance_total
        ratio = comment_freq / stance_freq
        if ratio > 1.5:
            comment_distinctive.append((token, ratio, count))
    for token, ratio, count in sorted(comment_distinctive, key=lambda x: -x[1])[:n]:
        print(f"  {token}: {ratio:.2f}x (count={count})")

# =============================================================================
# Part (b): LDA Topic Modeling
# =============================================================================

def run_lda(texts, n_topics=5, n_words=10):
    """Run LDA on a list of texts. Returns model, vectorizer, and feature names."""
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(texts)
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=RAND_SEED, max_iter=20)
    lda.fit(doc_term_matrix)
    
    feature_names = vectorizer.get_feature_names_out()
    return lda, vectorizer, feature_names

def get_topic_words(lda, feature_names, n_words=10):
    """Extract top words for each topic."""
    topics = []
    for topic in lda.components_:
        top_indices = topic.argsort()[:-n_words-1:-1]
        top_words = [(feature_names[i], topic[i]) for i in top_indices]
        topics.append(top_words)
    return topics

def print_topics(topics, title):
    """Print topic word lists."""
    print(f"\n{'='*80}")
    print(f"LDA TOPICS: {title}")
    print(f"{'='*80}")
    for i, topic in enumerate(topics):
        words = [w for w, _ in topic]
        print(f"Topic {i+1}: {', '.join(words)}")

def create_wordcloud(topics, title, save_path=None):
    """Create word cloud visualization for topics. Uses top word as topic name."""
    fig, axes = plt.subplots(1, len(topics), figsize=(4*len(topics), 4))
    if len(topics) == 1:
        axes = [axes]
    
    for ax, (i, topic) in zip(axes, enumerate(topics)):
        word_freq = {word: score for word, score in topic}
        wc = WordCloud(width=400, height=400, background_color='white',
                      colormap='viridis', max_words=50)
        wc.generate_from_frequencies(word_freq)
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(f'Topic {i+1}')
        ax.axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

def run_lda_analysis(df, n_topics=5):
    """Run LDA analysis on stance vs comment replies."""
    # Split data
    stance_df = df[df['label_text'].isin(['support', 'deny', 'query'])]
    comment_df = df[df['label_text'] == 'comment']
    
    print(f"\nStance samples (S/D/Q): {len(stance_df)}")
    print(f"Comment samples: {len(comment_df)}")
    
    # Preprocess
    stance_texts = [preprocess_for_lda(text) for text in stance_df['reply_text']]
    comment_texts = [preprocess_for_lda(text) for text in comment_df['reply_text']]
    
    # Run LDA on stance replies
    print("\nRunning LDA on Stance (S/D/Q) replies...")
    stance_lda, _, stance_features = run_lda(stance_texts, n_topics=n_topics)
    stance_topics = get_topic_words(stance_lda, stance_features)
    print_topics(stance_topics, "Stance Replies (S/D/Q)")
    filename = SAVE_DIR + "stance_wordcloud.png"
    create_wordcloud(stance_topics, "Stance Replies (S/D/Q) Topics", filename)
    
    # Run LDA on comment replies
    print("\nRunning LDA on Comment replies...")
    comment_lda, _, comment_features = run_lda(comment_texts, n_topics=n_topics)
    comment_topics = get_topic_words(comment_lda, comment_features)
    print_topics(comment_topics, "Comment Replies")
    filename = SAVE_DIR + "comment_wordcloud.png"
    create_wordcloud(comment_topics, "Comment Replies Topics", filename)

# %%
if __name__ == '__main__':
    print("Loading data...")
    all_df = get_all_data() # TODO analyse differenced between different datasets (train, dev, test)
    print(f"Total samples loaded: {len(all_df)}")
    
    # Part (a): Unigrams and bigrams by stance
    print("Part (a): Computing unigrams and bigrams by stance...")
    unigrams, bigrams = get_ngrams_by_stance(all_df)
    plot_ngrams_by_stance(unigrams, bigrams, top_n=10)
    
    # don't run below for now
    # Part (a): Token distribution comparison
    print("\nComparing stance vs non-stance token distributions...")
    distributions = compare_stance_vs_comment(all_df)
    print_distribution_comparison(distributions, n=15)
    
    # Part (b): LDA analysis
    print("\n" + "="*80)
    print("Part (b): LDA Topic Modeling")
    print("="*80)
    run_lda_analysis(all_df, n_topics=5)

# %%
