"""
GraphShield AI
XGBoost Fraud Detection Model
"""

from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
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

from xgboost import XGBClassifier


class XGBoostModel:

    @staticmethod
    def train(training_data):

        print("\n" + "=" * 60)
        print("XGBOOST FRAUD DETECTION MODEL")
        print("=" * 60)

        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Train-Test Split
        # ----------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        # ----------------------------------------------------
        # Feature Processing
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Handle Class Imbalance
        # ----------------------------------------------------

        scale_pos_weight = (
            (y_train == 0).sum() /
            max((y_train == 1).sum(), 1)
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        model = Pipeline([
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos_weight
                )
            )
        ])

        print("\nTraining XGBoost...")

        model.fit(X_train, y_train)

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        print("\nClassification Report\n")

        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0
            )
        )

        print("\nROC AUC Score")
        print(
            roc_auc_score(
                y_test,
                probabilities
            )
        )

        print("\nConfusion Matrix\n")

        print(
            confusion_matrix(
                y_test,
                predictions
            )
        )

        # ----------------------------------------------------
        # Save Model
        # ----------------------------------------------------

        output = Path("outputs/models")
        output.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
            model,
            output / "xgboost.pkl"
        )

        print("\nXGBoost Model Saved Successfully.")

        # ----------------------------------------------------
        # Return everything needed for SHAP
        # ----------------------------------------------------

        return {
            "metrics": {
                "Model": "XGBoost",
                "Accuracy": accuracy_score(
                    y_test,
                    predictions
                ),
                "Precision": precision_score(
                    y_test,
                    predictions,
                    zero_division=0
                ),
                "Recall": recall_score(
                    y_test,
                    predictions,
                    zero_division=0
                ),
                "F1": f1_score(
                    y_test,
                    predictions,
                    zero_division=0
                ),
                "ROC_AUC": roc_auc_score(
                    y_test,
                    probabilities
                )
            },
            "model": model,
            "X_test": X_test,
            "y_test": y_test,
            "predictions": predictions,
            "probabilities": probabilities
        }