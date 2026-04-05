"""Frequency pipeline with CBOW and TF-IDF."""

import sys
from pathlib import Path
import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import FREQ_METHODS, FREQ_MODELS_DIR, RAW_DATA_DIR, RESULTS_DIR
from src.data_download import download_and_save_data
from src.data_loader import DataLoader
from src.embeddings.frequency import build_frequency_logreg
from src.evaluation.metrics import evaluate_and_save_metrics


def run_pipeline() -> None:
    """Runs frequency-based experiments.

    Trains logistic regression for CBOW and TF-IDF.
    """
    if not (RAW_DATA_DIR / "train.csv").exists():
        download_and_save_data()

    loader = DataLoader()
    loader.process_and_save()

    x_train, y_train = loader.load_processed_data("train")
    x_test, y_test = loader.load_processed_data("test")

    rows: list[dict[str, float]] = []
    for method in FREQ_METHODS:
        print(f"Running {method}...")
        model = build_frequency_logreg(method=method)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        rows.append(evaluate_and_save_metrics(y_test, y_pred, f"LogReg_{method}"))
        joblib.dump(model, FREQ_MODELS_DIR / f"logreg_{method}.joblib")

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "module1_summary.csv", index=False)


if __name__ == "__main__":
    run_pipeline()
