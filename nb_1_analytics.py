# %%
import html
import re
import string
from collections import Counter # like dict except returns 0 rather than key error
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy

from data_loader import get_train_data, get_dev_data, get_test_data, get_all_data

# spacy
nlp = spacy.load('en_core_web_sm')

# global var
RAND_SEED = 42
STOPWORDS = nlp.Defaults.stop_words
PROTECTED_WORDS = {'isis', 'news', 'texas', 'paris'} # words not to lemmatize
SPECIFIC_TRANSFORMATIONS = { # special cases for normalisation (found empirically)
    r'\bcharlie\s+hebdo\b': 'charliehebdo',
    r'\bsydneysiege\b': 'sydney siege',
    r'\bgerman\s+wings\b': 'germanwings',
}
SAVE_DIR = './results/analytics/'

# use regex to remove patterns
RE_PATTERNS = [
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

# %%
# functions


# ---- preprocessing ----

def tokenize(text, lemmatize=True):
    """Tokenize: lowercase, decode HTML entities, remove URLs, mentions, punctuation, stopwords, and optionally lemmatize."""
    text = text.lower()
    
    # decode HTML entities (&amp, etc.)
    text = html.unescape(text)
    
    # remove patterns w/ regex
    for pattern in RE_PATTERNS:
        text = pattern.sub(r'', text)
    
    # remove punct
    punct_table = str.maketrans('', '', string.punctuation)
    text = text.translate(punct_table)
    
    # merge charlie hebdo (it appeared as 'charlie', 'hebdo' and 'charliehebdo')
    for pattern, replacement in SPECIFIC_TRANSFORMATIONS.items():
        text = re.sub(pattern, replacement, text)
    
    # tokenise, filter stopwords & lemmatise
    doc = nlp(text)
    if lemmatize:
        # protect certain words - otherwise lemmatise
        tokens = [
            t.text if t.text in PROTECTED_WORDS else t.lemma_
            for t in doc
            if t.text not in STOPWORDS and len(t.text) > 2 and not t.is_space
        ]
    else:
        # without lemmatisation for comparison
        tokens = [
            t.text for t in doc
            if t.text not in STOPWORDS and len(t.text) > 2 and not t.is_space
        ]
    
    return tokens


def get_bigrams(tokens):
    """Generate bigrams from a list of tokens."""
    return [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]


# ----- 1a unigrams and bigrams -----

def get_ngrams_by_stance(df):
    """Compute unigrams and bigrams for replies, grouped by stance label."""
    unigrams = {'support': Counter(), 'deny': Counter(), 'query': Counter(), 'comment': Counter()}
    bigrams = {'support': Counter(), 'deny': Counter(), 'query': Counter(), 'comment': Counter()}
    
    def process_row(row):
        tokens = tokenize(row['reply_text'])
        unigrams[row['label_text']].update(tokens)
        bigrams[row['label_text']].update(get_bigrams(tokens))
    
    df.apply(process_row, axis=1)
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


# ----- 1a comparing token dist -----

def compare_stance_vs_comment(df):
    """Compare token distributions: Stance (S/D/Q) vs Non-stance (Comment)."""
    stance_reply = Counter()
    comment_reply = Counter()
    counts = {'stance': 0, 'comment': 0}
    
    def process_row(row):
        tokens = tokenize(row['reply_text'])
        if row['label_text'] in ['support', 'deny', 'query']:
            stance_reply.update(tokens)
            counts['stance'] += 1
        else:
            comment_reply.update(tokens)
            counts['comment'] += 1
    
    df.apply(process_row, axis=1)
    return {
        'stance': {'reply': stance_reply, 'count': counts['stance']},
        'comment': {'reply': comment_reply, 'count': counts['comment']}
    }

def plot_distribution_comparison(distributions, n=15, save_dir=SAVE_DIR):
    """Plot comparison of token distributions using seaborn."""
    stance = distributions['stance']
    comment = distributions['comment']
    
    # Normalize for fair comparison
    stance_total = sum(stance['reply'].values())
    comment_total = sum(comment['reply'].values())
    
    # Find distinctive tokens for stance
    stance_distinctive = []
    for token, count in stance['reply'].most_common(100):
        stance_freq = count / stance_total
        comment_freq = comment['reply'].get(token, 1) / comment_total
        ratio = stance_freq / comment_freq
        if ratio > 1.5:
            stance_distinctive.append((token, ratio, count))
    stance_distinctive = sorted(stance_distinctive, key=lambda x: -x[1])[:n]
    
    # Find distinctive tokens for comment
    comment_distinctive = []
    for token, count in comment['reply'].most_common(100):
        comment_freq = count / comment_total
        stance_freq = stance['reply'].get(token, 1) / stance_total
        ratio = comment_freq / stance_freq
        if ratio > 1.5:
            comment_distinctive.append((token, ratio, count))
    comment_distinctive = sorted(comment_distinctive, key=lambda x: -x[1])[:n]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot stance-distinctive tokens
    if stance_distinctive:
        tokens, ratios, counts = zip(*stance_distinctive)
        sns.barplot(x=list(ratios), y=list(tokens), ax=axes[0], 
                   hue=list(tokens), palette='Blues_r', legend=False)
        axes[0].set_xlabel('Frequency Ratio (Stance / Comment)')
        axes[0].set_ylabel('Token')
        axes[0].set_title(f'Tokens MORE Common in STANCE (S/D/Q) Replies\n(Sample count: {stance["count"]})')
        axes[0].axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
    
    # Plot comment-distinctive tokens
    if comment_distinctive:
        tokens, ratios, counts = zip(*comment_distinctive)
        sns.barplot(x=list(ratios), y=list(tokens), ax=axes[1],
                   hue=list(tokens), palette='Oranges_r', legend=False)
        axes[1].set_xlabel('Frequency Ratio (Comment / Stance)')
        axes[1].set_ylabel('Token')
        axes[1].set_title(f'Tokens MORE Common in COMMENT Replies\n(Sample count: {comment["count"]})')
        axes[1].axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
    
    plt.suptitle('Token Distribution: Stance (S/D/Q) vs Non-Stance (Comment)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir + 'token_distribution_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir}token_distribution_comparison.png")
    plt.show()


# ----- 1b LDA topics -----
def lda_tokenize(text):
    return ' '.join(tokenize(text)) # string for countvectorizer

def run_lda(texts, n_topics=5):
    vectorizer = CountVectorizer(ngram_range=(1, 3)) # alr remove stopwords in tokenisation
    doc_term_matrix = vectorizer.fit_transform(texts)
    
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=20, random_state=RAND_SEED)
    lda.fit(doc_term_matrix)
    return lda, vectorizer.get_feature_names_out()

def get_topic_words(lda, feature_names, n_words=10):
    topics = []
    for topic in lda.components_:
        top_indices = topic.argsort()[:-n_words-1:-1]
        top_words = [(feature_names[i], topic[i]) for i in top_indices]
        topics.append(top_words)
    return topics

def print_topics(topics, title):
    print(f"\n{'='*80}")
    print(f"LDA TOPICS: {title}")
    print(f"{'='*80}")
    for i, topic in enumerate(topics):
        words = [w for w, _ in topic]
        print(f"Topic {i}: {', '.join(words)}")

def create_wordcloud(topics, title, save_path=None):
    fig, axes = plt.subplots(1, len(topics), figsize=(4*len(topics), 4))
    if len(topics) == 1:
        axes = [axes]
    
    for ax, (i, topic) in zip(axes, enumerate(topics)):
        word_freq = {word: score for word, score in topic}
        wc = WordCloud(width=400, height=400, background_color='white',
                        max_words=50)
        wc.generate_from_frequencies(word_freq)
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(f'Topic {i}', fontsize=16)
        ax.axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

def run_lda_analysis(df, n_topics=5):
    # Split data
    stance_df = df[df['label_text'].isin(['support', 'deny', 'query'])]
    comment_df = df[df['label_text'] == 'comment']
    
    print(f"\nStance samples (S/D/Q): {len(stance_df)}")
    print(f"Comment samples: {len(comment_df)}")
    
    # Preprocess
    stance_texts = [lda_tokenize(text) for text in stance_df['reply_text']]
    comment_texts = [lda_tokenize(text) for text in comment_df['reply_text']]
    
    # Run LDA on stance replies
    print("\nRunning LDA on Stance (S/D/Q) replies...")
    stance_lda, stance_features = run_lda(stance_texts, n_topics=n_topics)
    stance_topics = get_topic_words(stance_lda, stance_features)
    # print_topics(stance_topics, "Stance Replies (S/D/Q)")
    filename = SAVE_DIR + "stance_wordcloud.png"
    create_wordcloud(stance_topics, "Stance Replies (S/D/Q) Topics", filename)
    
    # Run LDA on comment replies
    print("\nRunning LDA on Comment replies...")
    comment_lda, comment_features = run_lda(comment_texts, n_topics=n_topics)
    comment_topics = get_topic_words(comment_lda, comment_features)
    # print_topics(comment_topics, "Comment Replies")
    filename = SAVE_DIR + "comment_wordcloud.png"
    create_wordcloud(comment_topics, "Comment Replies Topics", filename)

# %%
if __name__ == '__main__':
    print("Loading data...")
    all_df = get_all_data() # TODO analyse differenced between different datasets (train, dev, test)
    print(f"Total samples loaded: {len(all_df)}")
    
    print("Part (a): Computing unigrams and bigrams by stance...")
    unigrams, bigrams = get_ngrams_by_stance(all_df)
    plot_ngrams_by_stance(unigrams, bigrams, top_n=10)
    
    print("\nComparing stance vs non-stance token distributions...")
    distributions = compare_stance_vs_comment(all_df)
    plot_distribution_comparison(distributions, n=15)
    
    print("\n" + "="*80)
    print("Part (b): LDA Topic Modeling")
    print("="*80)
    run_lda_analysis(all_df, n_topics=5)

# %%
