
# 🧠 BERT-based Bug Report Classification

This project implements a bug report classification system using BERT (via HuggingFace Transformers) and compares it with a baseline Naive Bayes + TF-IDF model. It was developed as part of the *Intelligent Software Engineering* coursework at the University of Birmingham.

---

## 📁 Project Structure

```
BERT/
├── BERT.py                             # BERT classification using Transformers
├── NB+TF-IDF.py                        # Baseline model: Naive Bayes + TF-IDF
├── Sensitivity Analysis of Parameters.py  # Hyperparameter sensitivity test
├── Title+Body.csv                      # Dataset with 'text' and 'sentiment' columns
│
├── requirements.txt                    # Python dependencies
├── requirements.pdf                    # PDF version of the dependency list
├── manual.pdf                          # Project usage guide
├── replication.pdf                     # Replication guide for the experiments
│
├── README for NB+TF-IDF.md             # Markdown guide for the NB model
├── datasets/                           # (Optional) Additional datasets
└── results/                            # Directory for model outputs
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YourUsername/BERT-bug-report-classification.git
cd BERT-bug-report-classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📄 Dataset Format

Ensure `Title+Body.csv` exists in the project root and includes:

- `text`: Combined title + body of the bug report
- `sentiment`: Integer label (e.g., 0 = negative, 1 = positive)

Missing values will be replaced with empty strings.

---

## 🚀 How to Run

### BERT Classification

```bash
python BERT.py
```

- Fine-tunes `bert-base-uncased`
- Performs multiple runs (default: 10)
- Saves average evaluation metrics to `pytorch_BERT.csv`

### Naive Bayes + TF-IDF

```bash
python "NB+TF-IDF.py"
```

### Sensitivity Analysis

```bash
python "Sensitivity Analysis of Parameters.py"
```

---

## 📊 Output

- Evaluation metrics: Accuracy, Precision, Recall, F1-score, AUC
- Output file: `pytorch_BERT.csv` or files under `results/`

---

## 📚 Documentation

- `manual.pdf`: How to use the project
- `replication.pdf`: Step-by-step experiment replication
- `requirements.pdf`: Dependency explanations

---

## 🧠 Credits

Created for the **Intelligent Software Engineering** module  
University of Birmingham

---

## 📬 Contact

For any questions or suggestions, feel free to contact: `your.email@example.com`
