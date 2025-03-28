
# 🧠 BERT-based Bug Report Classification

This project implements a bug report classification system using a fine-tuned BERT model (via HuggingFace Transformers) and compares it with a baseline Naive Bayes + TF-IDF model. A parameter sensitivity analysis is also included to study model performance under different training configurations.

Developed for the **Intelligent Software Engineering** module at the University of Birmingham.

---

## 📁 Project Structure

```
BERT/
├── BERT.py                             # BERT classifier using Transformers + PyTorch
├── NB+TF-IDF.py                        # Baseline model using Naive Bayes + TF-IDF
├── Sensitivity Analysis of Parameters.py  # Script for parameter sensitivity test
├── Title+Body.csv                      # Dataset with 'text' and 'sentiment' columns
│
├── requirements.txt                    # Dependency list (text)
├── requirements.pdf                    # Dependency explanation (PDF)
├── manual.pdf                          # General user manual
├── replication.pdf                     # Full replication guide
│
├── README for NB+TF-IDF.md             # Markdown guide for NB+TF-IDF model
├── datasets/                           # Additional datasets (if any)
└── results/                            # Outputs from all models
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

**Python ≥ 3.8 is recommended**

---

## 📄 Dataset Format

Make sure the file `Title+Body.csv` is located in the project root and includes:

- `text`: Combined bug report title and body
- `sentiment`: Integer label (0 or 1)

Empty values will automatically be filled with `''` during preprocessing.

---

## 🚀 How to Run

### Run BERT Classification

```bash
python BERT.py
```

- Fine-tunes `bert-base-uncased`
- Repeats training and evaluation (default = 10 times)
- Computes: Accuracy, Precision, Recall, F1, AUC
- Saves output to: `pytorch_BERT.csv`

### Run Naive Bayes + TF-IDF Baseline

```bash
python "NB+TF-IDF.py"
```

- Runs the same dataset using a classic TF-IDF + Naive Bayes model
- Output: `pytorch_NB.csv`

### Run Parameter Sensitivity Analysis

```bash
python "Sensitivity Analysis of Parameters.py"
```

- Tests various values for learning rate, batch size, epochs, and threshold
- Output: `pytorch_BERT_sensitivity.csv`

---

## 📊 Output Files

- `pytorch_BERT.csv`: BERT model evaluation (average of 10 runs)
- `pytorch_NB.csv`: Naive Bayes results
- `pytorch_BERT_sensitivity.csv`: Results from sensitivity testing

---

## 🔧 Configuration Options

You can modify parameters directly in `BERT.py`, such as:

```python
model_name = 'bert-base-uncased'  # You can change to 'bert-base-cased', etc.
REPEAT = 10                       # Number of repeated train/test splits
```

---

## 📦 Requirements

Installed via `requirements.txt`:

```
pandas==2.2.3
numpy==2.1.3
torch==2.5.1
scikit-learn==1.6.0
transformers==4.49.0
tokenizers==0.21.0
tqdm==4.67.1
nltk==3.9.1
```

---

## 🧪 Replication Instructions

To replicate all results as shown in the report:

1. Install dependencies  
2. Run all three scripts (`BERT.py`, `NB+TF-IDF.py`, and `Sensitivity Analysis of Parameters.py`)  
3. Ensure `Title+Body.csv` is formatted correctly and in the root folder  
4. Outputs will be automatically saved under the project folder

More details in `replication.pdf`.

---

## 🧠 Notes

- GPU is optional but recommended for BERT training
- You may further expand the project with:
  - Multi-class classification
  - Multilingual models (e.g., `bert-base-chinese`)
  - Web interface or deployment options

---

## 📬 Contact

If you have any questions or feedback, feel free to reach out:  
**Your Name** – `your.email@example.com`

---

Happy experimenting! 🧪
