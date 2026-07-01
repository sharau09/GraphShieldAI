"""
GraphShield AI
Exploratory Data Analysis (EDA)
"""

from pathlib import Path
import matplotlib.pyplot as plt


class EDA:

    @staticmethod
    def fraud_distribution(df):

        print("\nGenerating Fraud Distribution Chart...")

        output_dir = Path("outputs/graphs")
        output_dir.mkdir(parents=True, exist_ok=True)

        fraud_counts = df["isFraud"].value_counts()

        plt.figure(figsize=(6, 5))

        plt.bar(
            ["Normal", "Fraud"],
            fraud_counts.values
        )

        plt.title("Fraud vs Normal Transactions")
        plt.xlabel("Class")
        plt.ylabel("Count")

        plt.tight_layout()

        plt.savefig(output_dir / "fraud_distribution.png")

        plt.close()

        print("✓ fraud_distribution.png created")


    @staticmethod
    def transaction_type_distribution(df):

        print("\nGenerating Transaction Type Distribution...")

        output_dir = Path("outputs/graphs")

        plt.figure(figsize=(8, 5))

        df["type"].value_counts().plot(kind="bar")

        plt.title("Transaction Type Distribution")
        plt.xlabel("Transaction Type")
        plt.ylabel("Count")

        plt.tight_layout()

        plt.savefig(output_dir / "transaction_type_distribution.png")

        plt.close()

        print("✓ transaction_type_distribution.png created")