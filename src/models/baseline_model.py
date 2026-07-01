"""
GraphShield AI
Baseline Fraud Detection Model
"""

from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


class BaselineModel:

    @staticmethod
    def train(training_data):

        print("\n" + "=" * 60)
        print("BASELINE MACHINE LEARNING MODEL")
        print("=" * 60)

        # ======================================================
        # Features
        # ======================================================

        features = [
            "type",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "sender_degree",
            "sender_in_degree",
            "sender_out_degree",
            "sender_pagerank",
            "sender_betweenness",
            "sender_clustering"
        ]

        X = training_data[features]
        y = training_data["isFraud"]

        # ======================================================
        # Train/Test Split
        # ======================================================

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        # ======================================================
        # Preprocessing
        # ======================================================

        categorical_features = ["type"]

        numerical_features = [
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "sender_degree",
            "sender_in_degree",
            "sender_out_degree",
            "sender_pagerank",
            "sender_betweenness",
            "sender_clustering"
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_features
                ),
                (
                    "num",
                    "passthrough",
                    numerical_features
                )
            ]
        )

        # ======================================================
        # Random Forest Model
        # ======================================================

        model = Pipeline([
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ])

        print("\nTraining Random Forest...")

        model.fit(X_train, y_train)

        # ======================================================
        # Predictions
        # ======================================================

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        # ======================================================
        # Evaluation Metrics
        # ======================================================

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )
        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )
        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )
        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

        print("\nClassification Report\n")

        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0
            )
        )

        print("\nROC AUC Score")
        print(roc_auc)

        print("\nConfusion Matrix\n")
        print(confusion_matrix(y_test, predictions))

        # ======================================================
        # Save Model
        # ======================================================

        output = Path("outputs/models")
        output.mkdir(
            parents=True,
            exist_ok=True
        )

        model_path = output / "random_forest.pkl"

        joblib.dump(
            model,
            model_path
        )

        print("\n✓ Random Forest Model Saved")
        print(f"Location : {model_path}")

        # ======================================================
        # Return Everything
        # ======================================================

        return {

            "model": model,

            "X_train": X_train,
            "X_test": X_test,

            "y_train": y_train,
            "y_test": y_test,

            "predictions": predictions,
            "probabilities": probabilities,

            "metrics": {

                "Model": "Random Forest",

                "Accuracy": accuracy,

                "Precision": precision,

                "Recall": recall,

                "F1": f1,

                "ROC_AUC": roc_auc
            }

        }