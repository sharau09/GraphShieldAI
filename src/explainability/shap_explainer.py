"""
GraphShield AI
SHAP Explainability
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


class SHAPExplainer:

    @staticmethod
    def explain(model, X):

        print("\n" + "=" * 60)
        print("SHAP MODEL EXPLAINABILITY")
        print("=" * 60)

        # -------------------------------------------------
        # Load model if a file path is provided
        # -------------------------------------------------

        if isinstance(model, str):
            model = joblib.load(model)

        classifier = model.named_steps["classifier"]
        preprocessor = model.named_steps["preprocessor"]

        X_processed = preprocessor.transform(X)

        # -------------------------------------------------
        # SHAP
        # -------------------------------------------------

        explainer = shap.TreeExplainer(classifier)

        shap_values = explainer.shap_values(X_processed)

        # Binary classifier compatibility
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        output = Path("outputs/explainability")
        output.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------
        # Summary Plot
        # -------------------------------------------------

        plt.figure(figsize=(12, 6))

        shap.summary_plot(
            shap_values,
            X_processed,
            show=False
        )

        plt.tight_layout()

        plt.savefig(
            output / "shap_summary.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print("✓ shap_summary.png created")

        # -------------------------------------------------
        # Feature Importance Plot
        # -------------------------------------------------

        plt.figure(figsize=(12, 6))

        shap.summary_plot(
            shap_values,
            X_processed,
            plot_type="bar",
            show=False
        )

        plt.tight_layout()

        plt.savefig(
            output / "shap_bar.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print("✓ shap_bar.png created")

        # -------------------------------------------------
        # Feature Importance CSV
        # -------------------------------------------------

        feature_names = preprocessor.get_feature_names_out()

        importance = np.abs(shap_values).mean(axis=0)

        df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        })

        df = df.sort_values(
            "Importance",
            ascending=False
        )

        df.to_csv(
            output / "feature_importance.csv",
            index=False
        )

        print("✓ feature_importance.csv created")

        return df