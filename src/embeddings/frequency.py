"""Frequency features and logistic regression training."""

from pathlib import Path
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.config import MAX_FEATURES, RANDOM_STATE


class FrequencyEmbedder:
    """Builds CBOW or TF-IDF vectorizers."""

    def __init__(self, method: str = "tfidf") -> None:
        """Initializes vectorizer.

        Args:
            method: One of cbow, bow, or tfidf.

        Raises:
            ValueError: If method is unsupported.
        """
        normalized = "cbow" if method == "bow" else method
        if normalized not in {"cbow", "tfidf"}:
            raise ValueError("method must be one of: cbow, bow, tfidf")
        self.method = normalized
        if normalized == "tfidf":
            self.vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words="english")
        else:
            self.vectorizer = CountVectorizer(max_features=MAX_FEATURES, stop_words="english")

    def fit_transform(self, corpus: list[str]):
        """Fits and transforms texts.

        Args:
            corpus: Input texts.

        Returns:
            Sparse feature matrix.
        """
        return self.vectorizer.fit_transform(corpus)

    def transform(self, corpus: list[str]):
        """Transforms texts using fitted vectorizer.

        Args:
            corpus: Input texts.

        Returns:
            Sparse feature matrix.
        """
        return self.vectorizer.transform(corpus)

    def save(self, filepath: str | Path) -> None:
        """Saves vectorizer to disk.

        Args:
            filepath: Destination path.
        """
        joblib.dump(self.vectorizer, Path(filepath))


def build_frequency_logreg(method: str) -> Pipeline:
    """Builds a frequency feature + logistic regression pipeline.

    Args:
        method: One of cbow, bow, or tfidf.

    Returns:
        Sklearn pipeline.
    """
    normalized = "cbow" if method == "bow" else method
    if normalized == "tfidf":
        vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words="english")
    elif normalized == "cbow":
        vectorizer = CountVectorizer(max_features=MAX_FEATURES, stop_words="english")
    else:
        raise ValueError("method must be one of: cbow, bow, tfidf")
    return Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
