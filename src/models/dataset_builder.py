"""
GraphShield AI
Training Dataset Builder
"""

from pathlib import Path
import pandas as pd


class DatasetBuilder:

    @staticmethod
    def build(sample_df, graph_features):

        print("\n" + "=" * 60)
        print("BUILDING TRAINING DATASET")
        print("=" * 60)

        sender_features = graph_features.add_prefix("sender_")

        training_data = sample_df.merge(
            sender_features,
            left_on="nameOrig",
            right_on="sender_customer",
            how="left"
        )

        training_data.drop(
            columns=["sender_customer"],
            inplace=True
        )

        output = Path("outputs/reports")
        output.mkdir(parents=True, exist_ok=True)

        training_data.to_csv(
            output / "training_dataset.csv",
            index=False
        )

        print(f"Training Records : {len(training_data):,}")

        print(training_data.head())

        return training_data