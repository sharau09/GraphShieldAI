"""
GraphShield AI
Model Comparison
"""

import pandas as pd
from pathlib import Path


class ModelComparison:

    @staticmethod
    def save(results):

        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        comparison = pd.DataFrame(results)

        print(comparison)

        output = Path("outputs/reports")
        output.mkdir(parents=True, exist_ok=True)

        comparison.to_csv(
            output / "model_comparison.csv",
            index=False
        )

        print("\nSaved to outputs/reports/model_comparison.csv")

        return comparison