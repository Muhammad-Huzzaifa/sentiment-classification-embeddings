"""Data loading and preprocessing."""

import re
import pandas as pd
import numpy as np
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


class DataLoader:
    """Loads and preprocesses dataset splits."""

    @staticmethod
    def clean_tweet(text: str) -> str:
        """Cleans tweet text.

        Args:
            text: Raw tweet text.

        Returns:
            Cleaned tweet text.
        """
        text = str(text).lower()
        text = re.sub(r'@user', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def process_and_save(self) -> None:
        """Processes raw splits and saves cleaned CSV files.

        Raises:
            FileNotFoundError: If raw files are missing.
        """
        print("Cleaning text data...")

        for split in ['train', 'validation', 'test']:
            input_path = RAW_DATA_DIR / f"{split}.csv"
            output_path = PROCESSED_DATA_DIR / f"{split}_clean.csv"

            if not input_path.exists():
                raise FileNotFoundError(f"Missing {input_path}. Run data_download.py first.")
            df = pd.read_csv(input_path)
            df['clean_text'] = df['text'].apply(self.clean_tweet)
            df = df[df['clean_text'] != '']
            df.to_csv(output_path, index=False)

            print(f"Processed {split:12} → {len(df):6} samples saved to {output_path.name}")

    def load_processed_data(self, split: str = 'train') -> tuple[np.ndarray, np.ndarray]:
        """Loads processed split.

        Args:
            split: One of train, validation, or test.

        Returns:
            Tuple of texts and labels.

        Raises:
            FileNotFoundError: If processed file is missing.
        """
        path = PROCESSED_DATA_DIR / f"{split}_clean.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Processed data not found at {path}. "
                f"Run DataLoader().process_and_save() first."
            )

        df = pd.read_csv(path).dropna()
        return df['clean_text'].values, df['label'].values

    @staticmethod
    def load_raw_data(split: str = 'train') -> pd.DataFrame:
        """Loads raw split.

        Args:
            split: One of train, validation, or test.

        Returns:
            DataFrame with raw rows.
        """
        path = RAW_DATA_DIR / f"{split}.csv"
        return pd.read_csv(path)
