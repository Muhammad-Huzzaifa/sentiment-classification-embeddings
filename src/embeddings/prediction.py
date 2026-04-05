"""Pretrained prediction-based embeddings."""

from typing import Iterable
import numpy as np
import gensim.downloader as api


class PretrainedWordEmbedder:
    """Loads pretrained Word2Vec or GloVe vectors."""

    def __init__(self, model_name: str) -> None:
        """Initializes embedder.

        Args:
            model_name: Gensim downloader model name.
        """
        self.model_name = model_name
        self.model = api.load(model_name)
        self.dim = int(self.model.vector_size)

    def sentence_matrix(self, text: str, max_seq_len: int) -> np.ndarray:
        """Builds fixed-length token embedding matrix.

        Args:
            text: Input sentence.
            max_seq_len: Max sequence length.

        Returns:
            Matrix shaped (max_seq_len, embedding_dim).
        """
        tokens = str(text).split()[:max_seq_len]
        rows = [self.model[token] for token in tokens if token in self.model]
        if not rows:
            return np.zeros((max_seq_len, self.dim), dtype=np.float16)
        arr = np.asarray(rows, dtype=np.float32)
        if arr.shape[0] < max_seq_len:
            padding = np.zeros((max_seq_len - arr.shape[0], self.dim), dtype=np.float32)
            arr = np.vstack([arr, padding])
        else:
            arr = arr[:max_seq_len]
        return arr.astype(np.float16, copy=False)

    def batch_sentence_matrix(self, texts: Iterable[str], max_seq_len: int) -> np.ndarray:
        """Builds fixed-length matrices for many texts.

        Args:
            texts: Input texts.
            max_seq_len: Max sequence length.

        Returns:
            Tensor-like array shaped (n, max_seq_len, embedding_dim).
        """
        return np.asarray(
            [self.sentence_matrix(text, max_seq_len=max_seq_len) for text in texts],
            dtype=np.float16,
        )
