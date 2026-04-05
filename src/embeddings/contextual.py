"""Pretrained contextual embeddings."""

import torch
import numpy as np
from typing import Optional
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


class ContextualEmbedder:
    """Base wrapper for transformer embeddings."""

    def __init__(self, model_name: str) -> None:
        """Initializes tokenizer and model.

        Args:
            model_name: HuggingFace model name.
        """
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def check_polysemy(self, sentence1: str, sentence2: str, target_word: str) -> Optional[float]:
        """Compares a word embedding across two contexts.

        Args:
            sentence1: First sentence.
            sentence2: Second sentence.
            target_word: Shared word to compare.

        Returns:
            Cosine similarity or None if token is not found.
        """
        print(f"\n--- Polysemy Check for the word: '{target_word}' ---")

        def get_word_vector(sentence: str) -> Optional[np.ndarray]:
            """Extracts one token vector from a sentence."""
            inputs = self.tokenizer(sentence, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            word_idx = -1
            for i, token in enumerate(tokens):
                if target_word.lower() in token.lower():
                    word_idx = i
                    break

            if word_idx == -1:
                print(f"Token '{target_word}' not found in: '{sentence}'")
                return None

            return outputs.last_hidden_state[0, word_idx, :].cpu().numpy()

        vec1 = get_word_vector(sentence1)
        vec2 = get_word_vector(sentence2)

        if vec1 is not None and vec2 is not None:
            similarity = cosine_similarity([vec1], [vec2])[0][0]
            print(f"Sentence 1: {sentence1}")
            print(f"Sentence 2: {sentence2}")
            print(f"Cosine Similarity: {similarity:.4f}")
            return similarity

        return None


class BERTEmbedder(ContextualEmbedder):
    """BERT sentence embedder."""

    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        """Initializes BERT embedder.

        Args:
            model_name: BERT model name.
        """
        super().__init__(model_name)

    def get_sentence_embeddings(self, texts: list[str], batch_size: int = 32, max_length: int = 128) -> np.ndarray:
        """Builds BERT sentence embeddings.

        Args:
            texts: Input texts.
            batch_size: Batch size.
            max_length: Max tokenizer length.

        Returns:
            Array shaped (n, hidden_size).
        """
        print(f"Extracting BERT embeddings in batches of {batch_size}...")
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                ).to(self.device)
                outputs = self.model(**inputs)

                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(cls_embeddings)

        return np.vstack(all_embeddings)


class GPTEmbedder(ContextualEmbedder):
    """GPT sentence embedder."""

    def __init__(self, model_name: str = "distilgpt2") -> None:
        """Initializes GPT embedder.

        Args:
            model_name: GPT model name.
        """
        super().__init__(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id

    def get_sentence_embeddings(self, texts: list[str], batch_size: int = 32, max_length: int = 128) -> np.ndarray:
        """Builds GPT sentence embeddings with mean pooling.

        Args:
            texts: Input texts.
            batch_size: Batch size.
            max_length: Max tokenizer length.

        Returns:
            Array shaped (n, hidden_size).
        """
        print(f"Extracting GPT embeddings in batches of {batch_size}...")
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                ).to(self.device)
                outputs = self.model(**inputs)

                attention_mask = (
                    inputs['attention_mask'].unsqueeze(-1)
                    .expand(outputs.last_hidden_state.size()).float()
                )
                sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask, 1)
                sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                mean_pooled = (sum_embeddings / sum_mask).cpu().numpy()

                all_embeddings.append(mean_pooled)

        return np.vstack(all_embeddings)
