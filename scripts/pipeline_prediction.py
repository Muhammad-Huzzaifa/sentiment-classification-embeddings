"""Prediction pipeline with pretrained GloVe and Word2Vec."""

import sys
from pathlib import Path
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    BATCH_SIZE,
    ENABLE_WORD2VEC,
    GLOVE_MODEL,
    LEARNING_RATE,
    MAX_SEQ_LEN,
    PRED_MODELS_DIR,
    RAW_DATA_DIR,
    RESULTS_DIR,
    NUM_CLASSES,
    RNN_EPOCHS,
    RNN_HIDDEN_DIM,
    RNN_LAYERS,
    WORD2VEC_MODEL,
)
from src.data_download import download_and_save_data
from src.data_loader import DataLoader
from src.embeddings.prediction import PretrainedWordEmbedder
from src.evaluation.metrics import evaluate_and_save_metrics
from src.models.classifiers import RNNClassifier
from src.models.train import create_data_loader, predict, train_model


def train_one_embedding(
    name: str,
    model_name: str,
    x_train_text: list[str],
    y_train,
    x_test_text: list[str],
    y_test,
) -> dict[str, float]:
    """Trains one pretrained embedding + RNN experiment.

    Args:
        name: Short display name.
        model_name: Gensim model name.
        x_train_text: Train texts.
        y_train: Train labels.
        x_test_text: Test texts.
        y_test: Test labels.

    Returns:
        Metric row.
    """
    embedder = PretrainedWordEmbedder(model_name=model_name)
    x_train = embedder.batch_sentence_matrix(x_train_text, max_seq_len=MAX_SEQ_LEN)
    x_test = embedder.batch_sentence_matrix(x_test_text, max_seq_len=MAX_SEQ_LEN)

    rnn = RNNClassifier(
        embedding_dim=embedder.dim,
        hidden_dim=RNN_HIDDEN_DIM,
        output_dim=NUM_CLASSES,
        num_layers=RNN_LAYERS,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = create_data_loader(x=x_train, y=y_train, batch_size=BATCH_SIZE, shuffle=True)
    train_model(
        model=rnn,
        train_loader=train_loader,
        epochs=RNN_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device,
    )

    y_pred = predict(model=rnn, x=x_test, batch_size=BATCH_SIZE, device=device)
    metrics = evaluate_and_save_metrics(y_test, y_pred, f"RNN_{name}")

    output_dir = PRED_MODELS_DIR / name.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(rnn.state_dict(), output_dir / "rnn.pt")
    return metrics


def run_pipeline() -> None:
    """Runs prediction-based experiments."""
    if not (RAW_DATA_DIR / "train.csv").exists():
        download_and_save_data()

    loader = DataLoader()
    loader.process_and_save()

    x_train_text, y_train = loader.load_processed_data("train")
    x_test_text, y_test = loader.load_processed_data("test")

    rows = [
        train_one_embedding(
            name="GloVe",
            model_name=GLOVE_MODEL,
            x_train_text=list(x_train_text),
            y_train=y_train,
            x_test_text=list(x_test_text),
            y_test=y_test,
        ),
    ]

    if ENABLE_WORD2VEC:
        try:
            rows.append(
                train_one_embedding(
                    name="Word2Vec",
                    model_name=WORD2VEC_MODEL,
                    x_train_text=list(x_train_text),
                    y_train=y_train,
                    x_test_text=list(x_test_text),
                    y_test=y_test,
                )
            )
        except (MemoryError, RuntimeError) as exc:
            print(f"Skipping Word2Vec because it does not fit in memory: {exc}")

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "module2_summary.csv", index=False)


if __name__ == "__main__":
    run_pipeline()
