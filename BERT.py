import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Configuration
project = 'pytorch'
datafile = 'Title+Body.csv'
REPEAT = 10
out_csv_name = f'{project}_BERT.csv'

# Load dataset
data = pd.read_csv(datafile).fillna('')

# Preprocess data
texts = data['text'].tolist()
labels = data['sentiment'].tolist()

# Define tokenizer
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

# Training & evaluation
results = []

for repeat_time in range(REPEAT):
    print(f'--- Repeat: {repeat_time+1}/{REPEAT} ---')

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=repeat_time
    )

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

    train_dataset = BugDataset(train_encodings, train_labels)
    test_dataset = BugDataset(test_encodings, test_labels)

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=5e-5,
        evaluation_strategy='epoch',
        logging_dir='./logs',
        logging_steps=50,
        save_strategy='no',
        seed=repeat_time
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    eval_result = trainer.evaluate()
    results.append(eval_result)

# Aggregate results
df_results = pd.DataFrame(results)
final_results = df_results.mean().to_dict()

print("\n=== BERT Classification Results ===")
print(f"Average Accuracy: {final_results['eval_accuracy']:.4f}")
print(f"Average Precision: {final_results['eval_precision']:.4f}")
print(f"Average Recall: {final_results['eval_recall']:.4f}")
print(f"Average F1 score: {final_results['eval_f1']:.4f}")
print(f"Average AUC: {final_results['eval_auc']:.4f}")

final_df = pd.DataFrame([{
    'repeated_times': REPEAT,
    'Accuracy': round(final_results['eval_accuracy'], 4),
    'Precision': round(final_results['eval_precision'], 4),
    'Recall': round(final_results['eval_recall'], 4),
    'F1': round(final_results['eval_f1'], 4),
    'AUC': round(final_results['eval_auc'], 4),
}])

final_df.to_csv(out_csv_name, index=False)

print(f"\nResults have been saved to: {out_csv_name}")

