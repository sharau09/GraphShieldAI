"""
GraphShield AI
Node Label Builder
"""

import torch


class LabelBuilder:

    @staticmethod
    def build(graph, transactions):

        print("\n" + "=" * 60)
        print("BUILDING NODE LABELS")
        print("=" * 60)

        node_mapping = {
            node: idx
            for idx, node in enumerate(graph.nodes())
        }

        labels = torch.zeros(len(node_mapping), dtype=torch.long)

        fraud_transactions = transactions[
            transactions["isFraud"] == 1
        ]

        fraud_customers = set(fraud_transactions["nameOrig"])

        for customer in fraud_customers:

            if customer in node_mapping:
                labels[node_mapping[customer]] = 1

        print(f"Fraud Nodes : {labels.sum().item()}")

        return labels