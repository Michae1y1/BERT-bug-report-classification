import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Configuration
project = 'pytorch'
datafile = 'Title+Body.csv'
out_csv_name = f'{project}_BERT_sensitivity.csv'

# Load dataset
data = pd.read_csv(datafile).fillna('')
texts = data['text'].tolist()
labels = data['sentiment'].tolist()

# Tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

class BugDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Metrics calculation
def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    f1 = f1_score(labels, predictions, average='macro')
    auc = roc_auc_score(labels, logits[:, 1])
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}

# Define parameter grid for sensitivity analysis
param_grid = {
    'learning_rate': [2e-5, 3e-5, 5e-5],
    'batch_size': [8, 16, 32],
    'epochs': [2, 3, 4],
    'threshold': [0.4, 0.5, 0.6]
}

results = []

# Sensitivity analysis
for param_name in param_grid:
    for val in param_grid[param_name]:
        print(f'\nTesting {param_name} = {val}')

        # Train/test split
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            data['text'], data['sentiment'], test_size=0.2, random_state=42
        )

        # Tokenization
        train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=512)
        test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, max_length=512)

        train_dataset = BugDataset(train_encodings, train_labels.tolist())
        test_dataset = BugDataset(test_encodings, test_labels.tolist())

        # Training arguments
        args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3 if param_name != 'epochs' else val,
            per_device_train_batch_size=8 if param_name != 'batch_size' else val,
            per_device_eval_batch_size=16,
            learning_rate=2e-5 if param_name != 'learning_rate' else val,
            evaluation_strategy='epoch',
            logging_steps=50,
            save_strategy='no',
            seed=42
        )

        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()

        # Predictions
        logits = trainer.predict(test_dataset).predictions
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]

        # Threshold sensitivity
        thresh = val if param_name == 'threshold' else 0.5
        preds = (probs >= thresh).astype(int)

        # Evaluate
        accuracy = accuracy_score(test_labels, preds)
        precision = precision_score(test_labels, preds, average='macro')
        recall = recall_score(test_labels, preds, average='macro')
        f1 = f1_score(test_labels, preds, average='macro')
        auc = roc_auc_score(test_labels, probs)

        # Save results
        result = {
            'parameter': param_name,
            'value': val,
            'threshold': thresh,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        results.append(result)

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(out_csv_name, index=False)
print(f'\nSensitivity analysis results saved to {out_csv_name}')
