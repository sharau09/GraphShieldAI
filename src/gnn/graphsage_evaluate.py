"""
GraphShield AI
GraphSAGE Evaluation
"""

import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


class GraphSAGEEvaluator:

    @staticmethod
    def evaluate(model, data):

        print("\n" + "=" * 60)
        print("GRAPHSAGE MODEL EVALUATION")
        print("=" * 60)

        model.eval()

        with torch.no_grad():

            out = model(data)

            probabilities = torch.softmax(out, dim=1)

            predictions = out.argmax(dim=1)

        y_true = data.y[data.test_mask].cpu().numpy()
        y_pred = predictions[data.test_mask].cpu().numpy()
        y_prob = probabilities[data.test_mask][:, 1].cpu().numpy()

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )
        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )
        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )
        roc_auc = roc_auc_score(
            y_true,
            y_prob
        )

        print(f"\nAccuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"ROC AUC   : {roc_auc:.4f}")

        print("\nClassification Report\n")

        print(
            classification_report(
                y_true,
                y_pred,
                zero_division=0
            )
        )

        print("\nConfusion Matrix\n")

        print(
            confusion_matrix(
                y_true,
                y_pred
            )
        )

        return {
            "Model": "GraphSAGE",
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc
        }