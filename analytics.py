import os
import argparse
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import numpy as np
from data_loader import RumourEvalDataset

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def get_top_ngrams(corpus, n=1, k=20):
    stop_words = set(stopwords.words('english'))
    # Add custom stopwords relevant to tweets (preprocessing should theoretically handle this, but adding safety)
    custom_stops = {'http', 'https', 'co', 'rt'}
    stop_words.update(custom_stops)
    
    vec = CountVectorizer(ngram_range=(n, n), stop_words=list(stop_words)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:k]

def plot_wordcloud(text_data, title, output_path):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(text_data))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def run_lda(text_data, num_topics=3, no_top_words=10):
    stop_words = list(set(stopwords.words('english')))
    vectorizer = CountVectorizer(stop_words=stop_words, max_df=0.95, min_df=2)
    dtm = vectorizer.fit_transform(text_data)
    
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(dtm)
    
    feature_names = vectorizer.get_feature_names_out()
    
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        topics[topic_idx] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        
    return topics

def analyze_dataset(data_path, label_path, output_dir):
    print(f"Loading data from {data_path}...")
    dataset = RumourEvalDataset(data_path, label_path)
    
    # Organize data by label
    data_by_label = {'support': [], 'deny': [], 'query': [], 'comment': []}
    source_texts = []
    
    for sample in dataset.samples:
        label = sample[2]
        reply_text = sample[1]
        source_text = sample[0]
        
        data_by_label[label].append(reply_text)
        source_texts.append(source_text) # This duplicates source texts, but fine for aggregate stats
        
    # unique source texts for cleaner analysis
    unique_source_texts = list(set(source_texts))

    # 1a: N-gram Analysis
    print("\n--- 1a: Top Unigrams and Bigrams ---")
    with open(os.path.join(output_dir, 'ngrams_summary.txt'), 'w') as f:
        for label, texts in data_by_label.items():
            f.write(f"\nLabel: {label.upper()} ({len(texts)} samples)\n")
            
            unigrams = get_top_ngrams(texts, n=1, k=10)
            f.write(f"Top Unigrams: {unigrams}\n")
            
            bigrams = get_top_ngrams(texts, n=2, k=10)
            f.write(f"Top Bigrams: {bigrams}\n")
            
            print(f"Computed n-grams for {label}")

    # Compare Token Distributions (Stance vs Non-Stance)
    print("\n--- Comparing Stance vs Non-Stance ---")
    stance_texts = data_by_label['support'] + data_by_label['deny'] + data_by_label['query']
    non_stance_texts = data_by_label['comment']
    
    # 1b: LDA Analysis
    print("\n--- 1b: LDA Topic Modeling ---")
    
    # LDA on Stance (S, D, Q)
    print("Running LDA on Stance replies...")
    stance_topics = run_lda(stance_texts, num_topics=3)
    with open(os.path.join(output_dir, 'lda_topics.txt'), 'w') as f:
        f.write("LDA Topics for STANCE (Support, Deny, Query):\n")
        for idx, words in stance_topics.items():
            f.write(f"Topic {idx}: {', '.join(words)}\n")
            
    plot_wordcloud(stance_texts, "Stance Replies Word Cloud", os.path.join(output_dir, 'stance_wordcloud.png'))

    # LDA on Comment (C)
    print("Running LDA on Comment replies...")
    comment_topics = run_lda(non_stance_texts, num_topics=3)
    with open(os.path.join(output_dir, 'lda_topics.txt'), 'a') as f:
        f.write("\nLDA Topics for NON-STANCE (Comment):\n")
        for idx, words in comment_topics.items():
            f.write(f"Topic {idx}: {', '.join(words)}\n")
            
    plot_wordcloud(non_stance_texts, "Comment Replies Word Cloud", os.path.join(output_dir, 'comment_wordcloud.png'))
    
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='results/analytics')
    args = parser.parse_args()
    
    train_data_root = 'data/semeval2017-task8-dataset/rumoureval-data'
    train_labels_path = 'data/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-train.json'
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    analyze_dataset(train_data_root, train_labels_path, args.output_dir)
