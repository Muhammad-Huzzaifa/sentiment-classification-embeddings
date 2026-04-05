"""Contextual pipeline with pretrained BERT and GPT."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    BATCH_SIZE,
    BERT_MODEL,
    CONTEXT_MODELS_DIR,
    GPT_MODEL,
    LEARNING_RATE,
    RAW_DATA_DIR,
    RESULTS_DIR,
    RNN_EPOCHS,
    RNN_HIDDEN_DIM,
    RNN_LAYERS,
)
from src.data_download import download_and_save_data
from src.data_loader import DataLoader
from src.embeddings.contextual import BERTEmbedder, GPTEmbedder
from src.evaluation.metrics import evaluate_and_save_metrics
from src.models.classifiers import RNNClassifier
from src.models.train import create_data_loader, predict, train_model


def train_context_model(
    name: str,
    x_train_seq: np.ndarray,
    y_train,
    x_test_seq: np.ndarray,
    y_test,
) -> dict[str, float]:
    """Trains RNN on contextual embeddings.

    Args:
        name: Model name label.
        x_train_seq: Train sequences.
        y_train: Train labels.
        x_test_seq: Test sequences.
        y_test: Test labels.

    Returns:
        Metric row.
    """
    rnn = RNNClassifier(
        embedding_dim=x_train_seq.shape[-1],
        hidden_dim=RNN_HIDDEN_DIM,
        output_dim=3,
        num_layers=RNN_LAYERS,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = create_data_loader(x=x_train_seq, y=y_train, batch_size=BATCH_SIZE, shuffle=True)
    train_model(
        model=rnn,
        train_loader=train_loader,
        epochs=RNN_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device,
    )
    y_pred = predict(model=rnn, x=x_test_seq, batch_size=BATCH_SIZE, device=device)
    metrics = evaluate_and_save_metrics(y_test, y_pred, f"RNN_{name}")
    output_dir = CONTEXT_MODELS_DIR / name.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(rnn.state_dict(), output_dir / "rnn.pt")
    return metrics


def run_pipeline() -> None:
    """Runs contextual embedding experiments."""
    if not (RAW_DATA_DIR / "train.csv").exists():
        download_and_save_data()

    loader = DataLoader()
    loader.process_and_save()

    x_train_text, y_train = loader.load_processed_data("train")
    x_test_text, y_test = loader.load_processed_data("test")

    bert = BERTEmbedder(model_name=BERT_MODEL)
    gpt = GPTEmbedder(model_name=GPT_MODEL)

    poly_bert = bert.check_polysemy(
        sentence1="I went to the bank to deposit money.",
        sentence2="I sat near the river bank.",
        target_word="bank",
    )
    poly_gpt = gpt.check_polysemy(
        sentence1="I went to the bank to deposit money.",
        sentence2="I sat near the river bank.",
        target_word="bank",
    )
    print(f"Polysemy BERT: {poly_bert}")
    print(f"Polysemy GPT: {poly_gpt}")

    x_train_bert = bert.get_sentence_embeddings(list(x_train_text))
    x_test_bert = bert.get_sentence_embeddings(list(x_test_text))
    x_train_gpt = gpt.get_sentence_embeddings(list(x_train_text))
    x_test_gpt = gpt.get_sentence_embeddings(list(x_test_text))

    x_train_bert_seq = np.expand_dims(x_train_bert, axis=1)
    x_test_bert_seq = np.expand_dims(x_test_bert, axis=1)
    x_train_gpt_seq = np.expand_dims(x_train_gpt, axis=1)
    x_test_gpt_seq = np.expand_dims(x_test_gpt, axis=1)

    rows = [
        train_context_model("BERT", x_train_bert_seq, y_train, x_test_bert_seq, y_test),
        train_context_model("GPT", x_train_gpt_seq, y_train, x_test_gpt_seq, y_test),
    ]

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "module3_summary.csv", index=False)


if __name__ == "__main__":
    run_pipeline()
