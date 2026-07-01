"""
GraphShield AI
Fraud Ring Detection
"""

from pathlib import Path
import pandas as pd
import networkx as nx


class FraudRingDetector:

    @staticmethod
    def detect(G):

        print("\n" + "=" * 60)
        print("FRAUD RING DETECTION")
        print("=" * 60)

        communities = list(nx.weakly_connected_components(G))

        print(f"Total Communities : {len(communities):,}")

        suspicious = [
            c for c in communities
            if len(c) >= 4
        ]

        print(
            f"Suspicious Communities (>=4 Nodes): {len(suspicious):,}"
        )

        print("\nTop 10 Largest Communities\n")

        largest = sorted(
            suspicious,
            key=len,
            reverse=True
        )[:10]

        for i, community in enumerate(largest, start=1):

            print(
                f"Community {i:<2} Nodes : {len(community)}"
            )

        records = []

        for i, community in enumerate(
            suspicious,
            start=1
        ):

            records.append({
                "Community": i,
                "Nodes": len(community),
                "Members": ", ".join(sorted(community))
            })

        output = Path("outputs/reports")
        output.mkdir(
            parents=True,
            exist_ok=True
        )

        pd.DataFrame(records).to_csv(
            output / "fraud_rings.csv",
            index=False
        )

        print("\n✓ fraud_rings.csv saved")

        return suspicious