# 🧠 BERT-based Bug Report Classification

This project implements a bug report classification system using BERT (via HuggingFace Transformers) and compares it with a baseline Naive Bayes + TF-IDF model. It is part of the Intelligent Software Engineering lab coursework.

---

## 📁 Project Structure

BERT/ ├── BERT.py # Main script for BERT classification ├── NB+TF-IDF.py # Baseline Naive Bayes classifier ├── Sensitivity Analysis of Parameters.py # Hyperparameter sensitivity experiment ├── Title+Body.csv # Dataset: contains 'text' and 'sentiment' columns │ ├── requirements.txt # Python dependencies ├── requirements.pdf # PDF version of dependency explanation ├── manual.pdf # User manual (PDF) ├── replication.pdf # Step-by-step replication guide │ ├── README for NB+TF-IDF.md # Notes specific to Naive Bayes model ├── datasets/ # (Optional) Extra datasets └── results/ # Folder for saved outputs (CSV, logs, etc.)

yaml
复制
编辑

---

## 📦 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/YourUsername/BERT-bug-report-classification.git
cd BERT-bug-report-classification
Install dependencies:

bash
复制
编辑
pip install -r requirements.txt
📄 Dataset Format
The dataset Title+Body.csv must contain the following columns:

text: The full bug report content (title + body combined)

sentiment: Integer label (e.g., 0 = negative, 1 = positive)

Empty values will be automatically filled as blank ('').

🚀 How to Run
🔹 Run BERT Classification
bash
复制
编辑
python BERT.py
Fine-tunes bert-base-uncased model

Repeats training/evaluation N times (default = 10)

Saves average results to pytorch_BERT.csv

🔹 Run Naive Bayes + TF-IDF Baseline
bash
复制
编辑
python "NB+TF-IDF.py"
🔹 Run Sensitivity Analysis
bash
复制
编辑
python "Sensitivity Analysis of Parameters.py"
📊 Output
Evaluation metrics: Accuracy, Precision, Recall, F1, AUC

Saved to results/ or directly as pytorch_BERT.csv

⚙️ Configurable Parameters
Edit BERT.py to modify:

Model name (e.g., bert-base-cased)

Training epochs, batch size, learning rate

Number of repetitions (REPEAT = 10)

📄 Documentation
manual.pdf – user guide

replication.pdf – how to replicate all results

requirements.pdf – dependency explanation

🧠 Credits
Developed as part of Intelligent Software Engineering coursework
University of Birmingham
