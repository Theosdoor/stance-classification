#!/usr/bin/env python3
"""
Export LDA visualizations as interactive HTML using pyLDAvis.
This script generates HTML visualizations for GitHub Pages at /LDA/
"""

import os
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.lda_model

# Suppress sklearn FutureWarnings about MDS parameters
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn.manifold')

from nb_1_analytics import lda_tokenize, RAND_SEED
from data_loader import get_all_data


def run_lda(texts, n_topics=5):
    """Run LDA and return model, vectorizer, and document-term matrix."""
    vectorizer = CountVectorizer(ngram_range=(1, 3))
    doc_term_matrix = vectorizer.fit_transform(texts)
    
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=20, random_state=RAND_SEED)
    lda.fit(doc_term_matrix)
    return lda, vectorizer, doc_term_matrix


def export_pyldavis(lda, vectorizer, doc_term_matrix, output_path):
    """Export interactive pyLDAvis visualization as HTML."""
    vis_data = pyLDAvis.lda_model.prepare(lda, doc_term_matrix, vectorizer, mds='mmds')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pyLDAvis.save_html(vis_data, output_path)
    print(f"Saved: {output_path}")


def create_index_page():
    """Create main LDA index page."""
    html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LDA Topic Analysis</title>
    <style>
        body { font-family: system-ui, sans-serif; background: #1a1a2e; color: #e7e9ea; padding: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { text-align: center; color: #00d9ff; }
        .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 30px; }
        .card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 24px; text-decoration: none; color: inherit; }
        .card:hover { background: rgba(255,255,255,0.1); }
        .card h2 { margin: 0 0 10px; }
        .card.stance h2 { color: #00ba7c; }
        .card.comment h2 { color: #ff7a00; }
        .card p { color: #71767b; margin: 0; }
        a { color: #1d9bf0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>LDA Topic Analysis</h1>
        <p style="text-align:center;color:#71767b;">Interactive topic visualization for SemEval rumor stance dataset</p>
        <div class="cards">
            <a href="stance/" class="card stance">
                <h2>Stance Topics (S/D/Q)</h2>
                <p>Topics from Support, Deny, Query tweets</p>
            </a>
            <a href="comment/" class="card comment">
                <h2>Comment Topics</h2>
                <p>Topics from neutral Comment tweets</p>
            </a>
        </div>
        <p style="margin-top:30px;"><a href="../">← Back</a></p>
        <footer style="margin-top:60px;padding-top:20px;border-top:1px solid rgba(255,255,255,0.1);text-align:center;color:#536471;font-size:0.8rem;">
            Created with the assistance of claude.ai (Anthropic)
        </footer>
    </div>
</body>
</html>'''
    os.makedirs('./LDA', exist_ok=True)
    with open('./LDA/index.html', 'w') as f:
        f.write(html)
    print("Saved: ./LDA/index.html")


def main():
    print("Loading data...")
    df = get_all_data()
    
    stance_df = df[df['label_text'].isin(['support', 'deny', 'query'])]
    comment_df = df[df['label_text'] == 'comment']
    
    print(f"Stance: {len(stance_df)}, Comment: {len(comment_df)}")
    
    stance_texts = [lda_tokenize(text) for text in stance_df['text']]
    comment_texts = [lda_tokenize(text) for text in comment_df['text']]
    
    print("Running LDA on stance...")
    stance_lda, stance_vec, stance_dtm = run_lda(stance_texts, n_topics=5)
    export_pyldavis(stance_lda, stance_vec, stance_dtm, './LDA/stance/index.html')
    
    print("Running LDA on comment...")
    comment_lda, comment_vec, comment_dtm = run_lda(comment_texts, n_topics=5)
    export_pyldavis(comment_lda, comment_vec, comment_dtm, './LDA/comment/index.html')
    
    create_index_page()
    print("\nDone! Open LDA/index.html to view.")


if __name__ == '__main__':
    main()
