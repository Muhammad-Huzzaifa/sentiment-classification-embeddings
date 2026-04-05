"""Dataset download helper."""

import pandas as pd
from datasets import load_dataset
from src.config import DATASET_NAME, DATASET_CONFIG, RAW_DATA_DIR


def download_and_save_data() -> None:
    """Downloads dataset and writes CSV files."""
    print(f"Downloading {DATASET_NAME} ({DATASET_CONFIG})...")
    print("This may take a moment on first run...\n")

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    for split in ['train', 'validation', 'test']:
        df = pd.DataFrame(dataset[split])
        output_path = RAW_DATA_DIR / f"{split}.csv"
        df.to_csv(output_path, index=False)
        print(f"✓ {split:12} data: {len(df):6} samples → {output_path.name}")

    print(f"\nAll data downloaded to: {RAW_DATA_DIR}")


if __name__ == "__main__":
    download_and_save_data()

