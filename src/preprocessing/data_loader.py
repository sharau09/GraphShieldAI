"""
GraphShield AI
Data Loader Module
"""

from pathlib import Path
import pandas as pd


class DataLoader:

    def __init__(self):
        self.dataset_path = Path(
            "data/raw/PS_20174392719_1491204439457_log.csv"
        )

    def load_dataset(self):

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found:\n{self.dataset_path}"
            )

        print("=" * 60)
        print("Loading PaySim Dataset...")
        print("=" * 60)

        df = pd.read_csv(self.dataset_path)

        print("Dataset loaded successfully.")
        print()

        return df