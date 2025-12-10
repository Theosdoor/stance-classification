import os
import torch
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from data_loader import RumourEvalDataset

from nb_main import CLASSIFIER_NAME

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Handle the num_items_in_batch argument which is new in recent transformers versions
        # Older versions don't pass it, so we need to be flexible or check signature
        # But Trainer.compute_loss signature is fixed in the base class. 
        # Actually, looking at recent Trainer code, num_items_in_batch is an argument.
        # But we can just capture it in **kwargs if we used *args, **kwargs in signature 
        # OR just define it in the signature.
        
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights.device != logits.device:
            self.class_weights = self.class_weights.to(logits.device)
            
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()

    # Paths
    train_data_root = 'data/semeval2017-task8-dataset/rumoureval-data'
    train_labels_path = 'data/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-train.json'
    dev_labels_path = 'data/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-dev.json'
    
    test_data_root = 'data/semeval2017-task8-test-data'
    test_labels_path = 'data/subtaska.json'

    # Load Data
    print("Loading datasets...")
    train_dataset = RumourEvalDataset(train_data_root, train_labels_path)
    dev_dataset = RumourEvalDataset(train_data_root, dev_labels_path) # Dev is in same folder structure but different labels
    test_dataset = RumourEvalDataset(test_data_root, test_labels_path)
    
    # Compute Class Weights
    train_labels = [sample[2] for sample in train_dataset.samples]
    unique_labels = np.array(sorted(list(set(train_labels))))
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=train_labels)
    # Map weights to ids: The ids are 0,1,2,3 corresponding to labels. 
    # RumourEvalDataset.label_to_id = {'support': 0, 'deny': 1, 'query': 2, 'comment': 3}
    # unique_labels should be ['comment', 'deny', 'query', 'support'] (sorted)
    # But we need to make sure the order matches the IDs.
    
    # Let's be explicit
    weights_map = dict(zip(unique_labels, class_weights))
    id_to_label = train_dataset.id_to_label
    ordered_weights = [weights_map[id_to_label[i]] for i in range(4)]
    
    print(f"Class Weights: {ordered_weights}")

    # Tokenizer
    model_name = CLASSIFIER_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["source_text"], examples["reply_text"], padding="max_length", truncation=True, max_length=128)

    # We need to process the dataset to apply tokenization
    # Since RumourEvalDataset yields dicts, we can wrap it or process it.
    # The standard way with Trainer is to have a dataset that returns dicts with 'input_ids', 'attention_mask', 'labels'.
    
    # Let's wrap the dataset or map it. 
    # Since it's a custom Torch dataset, we can just modify __getitem__ or use a collator.
    # But simpler is to convert it to a HF Dataset or pre-tokenize.
    # Given the small size, pre-tokenizing into memory is fine.
    
    def process_dataset(dataset):
        processed_data = []
        for i in range(len(dataset)):
            sample = dataset[i]
            tokenized = tokenizer(sample['source_text'], sample['reply_text'], padding='max_length', truncation=True, max_length=128)
            tokenized['labels'] = sample['label']
            processed_data.append(tokenized)
        return processed_data

    print("Tokenizing datasets...")
    train_encoded = process_dataset(train_dataset)
    dev_encoded = process_dataset(dev_dataset)
    test_encoded = process_dataset(test_dataset)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=f'{args.output_dir}/logs',
        logging_steps=100,
    )

    trainer = WeightedTrainer(
        class_weights=ordered_weights,
        model=model,
        args=training_args,
        train_dataset=train_encoded,
        eval_dataset=dev_encoded, # Use Dev for evaluation during training
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating on Test set...")
    test_results1 = trainer.predict(test_encoded)
    print("Test Results:", test_results1.metrics)
    
    # Confusion Matrix
    preds = np.argmax(test_results1.predictions, axis=1)
    labels = test_results1.label_ids
    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix:\n", cm)
    
    # Save confusion matrix plot? 
    # Maybe simpler to just print it for now.

if __name__ == "__main__":
    main()
