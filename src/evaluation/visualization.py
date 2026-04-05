"""Visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from src.config import RESULTS_DIR


def plot_pca_embeddings(
    words: List[str],
    vectors: np.ndarray,
    filename: str = "pca_embeddings.png"
) -> None:
    """Creates PCA scatter plot.

    Args:
        words: Labels for points.
        vectors: Embedding matrix.
        filename: Output image file name.
    """
    print("Generating PCA visualization...")

    pca = PCA(n_components=2, random_state=42)
    reduced_vectors = pca.fit_transform(vectors)

    plt.figure(figsize=(12, 10))
    plt.scatter(
        reduced_vectors[:, 0],
        reduced_vectors[:, 1],
        alpha=0.6,
        s=50,
        color='steelblue'
    )

    for i, word in enumerate(words):
        plt.annotate(
            word,
            (reduced_vectors[i, 0], reduced_vectors[i, 1]),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom',
            fontsize=9
        )

    explained_var = pca.explained_variance_ratio_
    plt.xlabel(f'PC1 ({explained_var[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({explained_var[1]*100:.1f}%)')
    plt.title('PCA Projection of Word Embeddings')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filepath = RESULTS_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"PCA plot saved to {filepath}")
    print(f"Variance explained: PC1={explained_var[0]:.2%}, PC2={explained_var[1]:.2%}")


def plot_tsne_embeddings(
    words: List[str],
    vectors: np.ndarray,
    filename: str = "tsne_embeddings.png",
    perplexity: int = 30
) -> None:
    """Creates t-SNE scatter plot.

    Args:
        words: Labels for points.
        vectors: Embedding matrix.
        filename: Output image file name.
        perplexity: t-SNE perplexity.
    """
    print("Generating t-SNE visualization...")

    perplexity = min(perplexity, len(words) - 1)
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        max_iter=1000
    )
    reduced_vectors = tsne.fit_transform(vectors)

    plt.figure(figsize=(12, 10))
    plt.scatter(
        reduced_vectors[:, 0],
        reduced_vectors[:, 1],
        alpha=0.6,
        s=50,
        color='darkblue'
    )

    for i, word in enumerate(words):
        plt.annotate(
            word,
            (reduced_vectors[i, 0], reduced_vectors[i, 1]),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom',
            fontsize=9
        )

    plt.title('t-SNE Projection of Word Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filepath = RESULTS_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"t-SNE plot saved to {filepath}")


def plot_embedding_comparison(
    words: List[str],
    embeddings_dict: dict[str, np.ndarray],
    filename: str = "embedding_comparison.png"
) -> None:
    """Creates side-by-side PCA plots.

    Args:
        words: Labels for points.
        embeddings_dict: Method name to vectors map.
        filename: Output image file name.
    """
    print("Generating embedding comparison visualization...")

    n_methods = len(embeddings_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(15, 5))

    if n_methods == 1:
        axes = [axes]

    for idx, (method_name, vectors) in enumerate(embeddings_dict.items()):
        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(vectors)

        axes[idx].scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=50)
        for i, word in enumerate(words):
            axes[idx].annotate(
                word,
                (reduced[i, 0], reduced[i, 1]),
                xytext=(3, 3),
                textcoords='offset points',
                fontsize=8
            )

        axes[idx].set_title(f'{method_name}')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = RESULTS_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Comparison plot saved to {filepath}")
