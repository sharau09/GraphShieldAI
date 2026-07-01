"""
GraphShield AI
Customer Risk Score
"""

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


class RiskScore:

    @staticmethod
    def generate(graph_features):

        print("\n" + "=" * 60)
        print("CUSTOMER RISK SCORE")
        print("=" * 60)

        # ------------------------------------------
        # Copy graph features
        # ------------------------------------------

        df = graph_features.copy()

        # ------------------------------------------
        # Normalize selected graph metrics
        # ------------------------------------------

        scaler = MinMaxScaler()

        columns = [
            "degree",
            "pagerank",
            "betweenness",
            "clustering"
        ]

        df[columns] = scaler.fit_transform(df[columns])

        # ------------------------------------------
        # Calculate Risk Score
        # ------------------------------------------

        df["risk_score"] = (
            0.35 * df["degree"] +
            0.30 * df["pagerank"] +
            0.25 * df["betweenness"] +
            0.10 * df["clustering"]
        )

        df = df.sort_values(
            by="risk_score",
            ascending=False
        )

        # ------------------------------------------
        # Keep ONLY required columns
        # ------------------------------------------

        risk_scores = df[
            [
                "customer",
                "risk_score"
            ]
        ].copy()

        # ------------------------------------------
        # Save Results
        # ------------------------------------------

        output = Path("outputs/reports")
        output.mkdir(
            parents=True,
            exist_ok=True
        )

        risk_scores.to_csv(
            output / "customer_risk_scores.csv",
            index=False
        )

        # ------------------------------------------
        # Display Top Customers
        # ------------------------------------------

        print("\nTop 10 Highest Risk Customers\n")
        print(risk_scores.head(10))

        return risk_scores