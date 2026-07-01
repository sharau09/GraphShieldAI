"""
GraphShield AI
Model Comparison
"""

from pathlib import Path
import pandas as pd


class ModelComparison:

    @staticmethod
    def save(results):

        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        comparison = pd.DataFrame(results)

        print(comparison)

        output_dir = Path("outputs/reports")
        output_dir.mkdir(parents=True, exist_ok=True)

        comparison.to_csv(
            output_dir / "model_comparison.csv",
            index=False
        )

        print("\nModel comparison saved successfully.")

        return comparison