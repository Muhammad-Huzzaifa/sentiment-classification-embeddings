# Sentiment Classification with Word Embeddings 📘

A comprehensive NLP assignment implementing and comparing three paradigms of word representations for Twitter sentiment classification.

## 📋 Project Overview

This project explores the evolution of word representations in NLP through three distinct approaches, each demonstrating different advantages:

### **Module 1: Frequency-Based Embeddings**
Traditional statistical approaches that count word occurrences
- **Techniques**: Bag of Words (BoW), TF-IDF
- **Classifier**: Logistic Regression
- **Advantages**: Simple, interpretable, low computational cost
- **Trade-offs**: Ignores word order and semantic relationships

### **Module 2: Prediction-Based Embeddings**
Neural methods that learn representations by predicting context
- **Techniques**: GloVe (pretrained), Word2Vec (pretrained via Gensim)
- **Classifier**: RNN (LSTM)
- **Advantages**: Captures semantic similarity, transferable representations
- **Trade-offs**: Single representation per word (no polysemy handling)

### **Module 3: Contextualized Embeddings**
Transformer-based methods that generate context-dependent representations
- **Techniques**: BERT, GPT (distilgpt2)
- **Classifier**: RNN (LSTM)
- **Advantages**: Context-aware, handles polysemy, state-of-the-art performance
- **Trade-offs**: Computationally expensive, requires GPU for efficiency

---

## 🗂️ Project Structure

```
sentiment-classification-embeddings/
├── src/                           # Main source code
│   ├── embeddings/               # Embedding implementations
│   │   ├── frequency.py          # BoW, TF-IDF
│   │   ├── prediction.py         # GloVe, Word2Vec
│   │   └── contextual.py         # BERT, GPT
│   ├── models/                   # Classification models
│   │   ├── classifiers.py        # SklearnClassifier, FNNClassifier, RNNClassifier
│   │   └── train.py              # Training utilities
│   ├── evaluation/               # Metrics and visualization
│   │   ├── metrics.py            # Evaluation metrics
│   │   └── visualization.py      # PCA, t-SNE, comparison plots
│   ├── config.py                 # Configuration settings
│   ├── data_loader.py            # Data loading and preprocessing
│   └── data_download.py          # Dataset download
├── scripts/                       # Pipeline scripts
│   ├── pipeline_frequency.py     # Module 1 pipeline
│   ├── pipeline_prediction.py    # Module 2 pipeline
│   └── pipeline_contextual.py    # Module 3 pipeline
├── notebooks/                    # Jupyter notebooks (interactive)
│   └── notebook.ipynb
├── interface/                    # Web interface (optional)
│   ├── app.py
│   └── static/
├── data/                         # Datasets
│   ├── raw/                      # Downloaded raw data
│   └── processed/                # Cleaned preprocessed data
├── models/                       # Trained models
│   ├── frequency/                # Module 1 models
│   ├── prediction/               # Module 2 models
│   └── contextual/               # Module 3 models
├── results/                      # Results and visualizations
│   ├── cm_*.png                  # Confusion matrices
│   ├── *_tsne.png                # t-SNE visualizations
│   ├── *_pca.png                 # PCA visualizations
│   └── module*_summary.csv       # Performance summaries
├── requirements.txt              # Python dependencies
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd sentiment-classification-embeddings

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 2. Run Individual Modules

**Module 1: Frequency-Based Embeddings**
```bash
python scripts/pipeline_frequency.py
```
- Trains BoW and TF-IDF with Logistic Regression
- Saves models and evaluation metrics
- Output: `results/module1_summary.csv`, confusion matrices

**Module 2: Prediction-Based Embeddings**
```bash
python scripts/pipeline_prediction.py
```
- Trains GloVe + RNN by default
- Optionally trains Word2Vec + RNN when `ENABLE_WORD2VEC = True` in `src/config.py`
- Compares embeddings (PCA and t-SNE)
- Output: `results/module2_summary.csv`, visualizations

**Module 3: Contextualized Embeddings**
```bash
python scripts/pipeline_contextual.py
```
- Extracts and trains BERT + RNN and GPT + RNN
- Includes polysemy analysis
- Output: `results/module3_summary.csv`, visualizations

### 3. Run All Modules

```bash
# Create a master script or run sequentially:
python scripts/pipeline_frequency.py && \
python scripts/pipeline_prediction.py && \
python scripts/pipeline_contextual.py
```

---

## 📊 Key Features

### Embedding Techniques Implemented

| Technique | Type | Training | Contextual | Polysemy |
|-----------|------|----------|-----------|----------|
| Bag of Words | Frequency | ✗ | ✗ | ✗ |
| TF-IDF | Frequency | ✗ | ✗ | ✗ |
| GloVe | Prediction (Pretrained) | ✗ | ✗ | ✗ |
| Word2Vec | Prediction (Pretrained) | ✗ | ✗ | ✗ |
| BERT | Contextual | ✓ | ✓ | ✓ |
| GPT | Contextual | ✓ | ✓ | ✓ |

### Classifiers

- **Logistic Regression**: Traditional ML baseline (Module 1)
- **Feed-Forward Neural Network**: Dense vector processing (not actively used)
- **Recurrent Neural Network**: Sequential embedding processing (Modules 2 & 3)
    - LSTM layers
  - Configurable hidden dimensions

### Evaluation Metrics

For each module:
- **Accuracy**: Overall correctness
- **Precision**: Positive prediction accuracy
- **Recall**: True positive detection rate
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrices**: Visual error analysis

### Visualizations

- **t-SNE**: Nonlinear dimensionality reduction (clusters similar words)
- **PCA**: Linear dimensionality reduction (variance preservation)
- **Embedding Comparison**: Side-by-side method comparison
- **Confusion Matrices**: Per-method classification breakdown

---

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
# Dataset settings
DATASET_NAME = "tweet_eval"
DATASET_CONFIG = "sentiment"
NUM_CLASSES = 3  # negative, neutral, positive

# Preprocessing
MAX_FEATURES = 5000  # BoW/TF-IDF vocabulary size
RANDOM_STATE = 42

# Embedding model names
GLOVE_MODEL = "glove-twitter-50"
WORD2VEC_MODEL = "word2vec-google-news-300"
ENABLE_WORD2VEC = False
BERT_MODEL = "bert-base-uncased"
GPT_MODEL = "distilgpt2"

# Training settings
MAX_SEQ_LEN = 64
BATCH_SIZE = 32
LEARNING_RATE = 0.001
RNN_HIDDEN_DIM = 128
RNN_LAYERS = 1
RNN_EPOCHS = 8
```

---

## 📈 Observed Results

These are the results from the current codebase and dataset split sizes.

### Module 1: Frequency-Based
- **BoW/CBOW + LogReg**: Accuracy 0.5724, Precision 0.5785, Recall 0.5724, F1 0.5625
- **TF-IDF + LogReg**: Accuracy 0.5732, Precision 0.5831, Recall 0.5732, F1 0.5610

### Module 2: Prediction-Based
- **GloVe + RNN**: Accuracy 0.5970, Precision 0.6283, Recall 0.5970, F1 0.5862
- **Word2Vec + RNN**: Accuracy 0.6224, Precision 0.6280, Recall 0.6224, F1 0.6219

### Module 3: Contextualized
- **BERT + RNN**: Accuracy 0.6012, Precision 0.6394, Recall 0.6012, F1 0.5912
- **GPT + RNN**: Accuracy 0.6178, Precision 0.6437, Recall 0.6178, F1 0.6110
- Polysemy check for "bank": BERT cosine similarity 0.5230, GPT cosine similarity 0.9940

Notes:
- The values above reflect the current artifacts in `results/` (including a run where Word2Vec was enabled).
- If `ENABLE_WORD2VEC = False`, rerunning Module 2 will only report GloVe.
- The BERT and GPT model-loading warnings about unexpected keys are normal for this architecture mismatch and can be ignored.
- The contextual pipeline is the slowest stage because it downloads transformer models and extracts embeddings for the full dataset.

---

## 📚 Code Quality Standards

This project follows **PEP 8** with enhancements:

✅ **Type Hints**: All functions have complete type annotations
✅ **NumPy Docstrings**: Every function has detailed documentation
✅ **Naming Conventions**: Descriptive `snake_case` for functions/variables
✅ **Line Length**: ≤ 88 characters (Black-compatible)
✅ **Comments**: Only when explaining "why", not "what"

Example:
```python
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 20,
    device: torch.device = torch.device("cpu")
) -> dict:
    """Train PyTorch model with validation.

    Parameters
    ----------
    model : nn.Module
        Model to train
    train_loader : torch.utils.data.DataLoader
        Training data loader
    epochs : int, default=20
        Number of training epochs
    device : torch.device, default=cpu
        Device to run on

    Returns
    -------
    dict
        Training history
    """
```

---

## 🔍 Assignment Requirements Checklist

### Module 1: Frequency-Based Embeddings
- [x] Preprocessing (tokenization, lowercasing, stopword removal, vocabulary)
- [x] Feature extraction (BoW, TF-IDF vectors)
- [x] Model training (Logistic Regression)
- [x] Evaluation (accuracy, precision, recall, F1)

### Module 2: Prediction-Based Embeddings
- [x] Embedding generation (GloVe pretrained + Word2Vec pretrained)
- [x] Multiple embedding dimensions
- [x] Model training (RNN classifier)
- [x] Evaluation (same metrics as Module 1)
- [x] Visualization (PCA and t-SNE)

### Module 3: Contextualized Embeddings
- [x] Embedding extraction (BERT, GPT from HuggingFace)
- [x] Model training (RNN classifier)
- [x] Polysemy check (demonstrate context-dependency)
- [x] Evaluation (same metrics as Modules 1 & 2)

---

## 💡 Key Insights

### Why Different Embeddings?

1. **Frequency-based**
   - Fast and simple
   - Good for quick baselines
   - Limited semantic understanding

2. **Prediction-based**
   - Better semantic relationships
   - Transferable across tasks
   - Still single representation per word

3. **Contextualized**
   - Different vectors for different contexts
   - Handles polysemy naturally
   - Requires more computational resources

### Polysemy Example

The word "bank" in:
1. "I went to the **bank** to deposit money" (financial institution)
2. "I sat by the river **bank**" (riverbank)

- **Word2Vec**: Same vector for both contexts
- **BERT/GPT**: Different vectors → low similarity score

---

## 🐛 Troubleshooting

### GPU Not Available
```python
# Models will automatically fall back to CPU
# To force CPU: modify scripts to set device = torch.device("cpu")
```

### Memory Issues
- Reduce `batch_size` in config
- Set `ENABLE_WORD2VEC = False` in `src/config.py` for low-memory environments
- Reduce `max_features` for BoW/TF-IDF

### Missing Data
```bash
# Data will be downloaded automatically on first run
# Or manually download:
python src/data_download.py
```

---

## 📖 References

### Papers
- Word2Vec: [Mikolov et al., 2013](https://arxiv.org/abs/1301.3781)
- GloVe: [Pennington et al., 2014](https://nlp.stanford.edu/pubs/glove.pdf)
- BERT: [Devlin et al., 2018](https://arxiv.org/abs/1810.04805)
- GPT: [Radford et al., 2018](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

### Datasets
- TweetEval: https://huggingface.co/datasets/tweet_eval

### Resources
- HuggingFace Transformers: https://huggingface.co/transformers/
- Gensim: https://radimrehurek.com/gensim/
- PyTorch: https://pytorch.org/
