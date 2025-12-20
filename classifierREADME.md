Stance Classifier with LoRA Fine-tuning
Implement a 4-way stance classifier (Support, Deny, Query, Comment) for RumourEval 2017 Subtask A using DeBERTa-v3-base with LoRA fine-tuning, incorporating W&B for hyperparameter optimization and diagnostic tracking.

Model Selection
DeBERTa-v3-base recommended over BERTweet for the following reasons:

Superior performance: DeBERTa-v3 achieves SOTA on SuperGLUE and many NLU benchmarks
Disentangled attention: Better captures relative position information, important for understanding reply context
Gradient-disentangled embedding sharing: More parameter-efficient fine-tuning
Better generalization: More robust to distribution shifts
NOTE

BERTweet (vinai/bertweet-base) could be used if Twitter-specific vocabulary is deemed critical. The architecture supports swapping models easily.

Proposed Changes
Core Module
[MODIFY] 
classifier.py
Complete rewrite implementing:

Model Architecture

Base: microsoft/deberta-v3-base from HuggingFace
LoRA adapter using PEFT library (rank=16, alpha=32, target modules: query/value projections)
Classification head: 4-way softmax
Input Format

Concatenation: [CLS] source_text [SEP] reply_text [SEP]
Max length: 256 tokens (social media posts are short)
Training Pipeline

AdamW optimizer with linear warmup scheduler
Class-weighted cross-entropy loss (inverse frequency weighting)
Early stopping based on validation macro-F1
W&B Integration

Hyperparameter sweep configuration
Automatic logging of loss curves and metrics
Model artifact saving
Configuration
[NEW] 
.env.example
Template for W&B API key:

WANDB_API_KEY=your_api_key_here
Dependencies
[MODIFY] 
requirements.txt
Add required packages:

torch
wandb
transformers
peft
datasets
scikit-learn
python-dotenv
tqdm
accelerate
Preprocessing Justification
Step	Rationale
No aggressive preprocessing	Pre-trained transformers have seen diverse web text; retain original tokens for model's vocabulary coverage
Source + Reply concatenation	Context is crucial—replies often reference source content implicitly; separator token marks boundary
Max length 256	Tweets are max 280 chars; 256 tokens provides ample coverage without padding overhead
HTML entity decoding	Standard practice for web-scraped text (handled by tokenizer implicitly)
IMPORTANT

Unlike traditional NLP pipelines, we avoid lemmatization, stopword removal, and lowercasing because pre-trained transformers use their own subword tokenizers and benefit from original case/form information.

Class Imbalance Mitigation
The RumourEval dataset exhibits severe class imbalance:

Comment: ~70% of samples (majority class)
Support/Deny/Query: ~30% combined (minority classes)
Strategies Implemented:

Weighted Cross-Entropy Loss

Compute class weights inversely proportional to class frequency
Formula: weight[c] = total_samples / (num_classes * count[c])
Stratified Sampling

Maintain class distribution in train/validation splits
Macro-F1 as Primary Metric

Equal importance to all classes regardless of frequency
NOTE

We avoid oversampling/SMOTE as they don't work well with pre-trained transformer embeddings and can cause overfitting on synthetic examples.

Hyperparameter Search Space (W&B Sweep)
Hyperparameter	Search Space	Rationale
Learning rate	[1e-5, 3e-5, 5e-5]	Standard range for transformer fine-tuning
LoRA rank	[8, 16, 32]	Balance model capacity vs. overfitting
LoRA alpha	[16, 32, 64]	Alpha = 2*rank is common heuristic
Batch size	[8, 16]	Memory constraints; smaller batches may regularize
Warmup ratio	[0.0, 0.1, 0.2]	Prevent early divergence
Weight decay	[0.0, 0.01, 0.1]	Regularization strength
Verification Plan
Automated Tests
Run training script: python classifier.py
Verify W&B dashboard shows:
Training loss curve (decreasing trend)
Validation macro-F1 per epoch (improving then plateau/overfit)
Confirm best model checkpoint saved
Manual Verification
Check W&B run page for logged metrics
Export diagnostic plots from W&B dashboard
Verify confusion matrix distribution across classes
Expected Outputs
classifier.py: Complete training pipeline with LoRA + W&B
.env.example: Template for API key configuration
W&B Dashboard:
Loss curves (train + validation)
Macro-F1 vs. epoch
Hyperparameter sweep results
Best model artifacts