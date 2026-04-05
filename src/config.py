"""Project configuration."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

FREQ_MODELS_DIR = MODELS_DIR / "frequency"
PRED_MODELS_DIR = MODELS_DIR / "prediction"
CONTEXT_MODELS_DIR = MODELS_DIR / "contextual"

for path in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FREQ_MODELS_DIR,
    PRED_MODELS_DIR,
    CONTEXT_MODELS_DIR,
    RESULTS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

DATASET_NAME = "tweet_eval"
DATASET_CONFIG = "sentiment"

RANDOM_STATE = 42
MAX_FEATURES = 5000
NUM_CLASSES = 3

FREQ_METHODS = ["cbow", "tfidf"]

GLOVE_MODEL = "glove-twitter-50"
WORD2VEC_MODEL = "word2vec-google-news-300"
ENABLE_WORD2VEC = False

BERT_MODEL = "bert-base-uncased"
GPT_MODEL = "distilgpt2"

MAX_SEQ_LEN = 64
BATCH_SIZE = 32
LEARNING_RATE = 0.001
RNN_HIDDEN_DIM = 128
RNN_LAYERS = 1
RNN_EPOCHS = 8
