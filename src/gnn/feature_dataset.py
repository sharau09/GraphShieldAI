"""
GraphShield AI
GNN Feature Dataset Builder
"""

import numpy as np
import torch
from torch_geometric.data import Data


class GNNFeatureDataset:

    @staticmethod
    def build(graph, customer_features):

        print("\n" + "=" * 60)
        print("BUILDING GNN FEATURE DATASET")
        print("=" * 60)

        # -------------------------------------------------
        # Create Node Mapping
        # -------------------------------------------------
        node_mapping = {
            node: idx
            for idx, node in enumerate(graph.nodes())
        }

        # -------------------------------------------------
        # Build Edge Index
        # -------------------------------------------------
        edge_index = []

        for u, v in graph.edges():
            edge_index.append([
                node_mapping[u],
                node_mapping[v]
            ])

        edge_index = torch.tensor(
            edge_index,
            dtype=torch.long
        ).t().contiguous()

        # -------------------------------------------------
        # Feature Columns
        # -------------------------------------------------
        feature_columns = [

            # Customer Behaviour Features
            "transaction_count",
            "total_sent_amount",
            "average_transaction_amount",
            "max_transaction_amount",
            "cash_out_count",
            "transfer_count",
            "payment_count",
            "fraud_ratio",
            "risk_score",

            # Graph Features
            "degree",
            "in_degree",
            "out_degree",
            "pagerank",
            "betweenness",
            "clustering"
        ]

        customer_features = customer_features.set_index("customer")

        # -------------------------------------------------
        # Build Node Feature Matrix
        # -------------------------------------------------
        x = []

        for node in graph.nodes():

            if node in customer_features.index:

                values = customer_features.loc[
                    node,
                    feature_columns
                ].fillna(0)

                x.append(
                    values.astype(np.float32).to_numpy()
                )

            else:

                x.append(
                    np.zeros(
                        len(feature_columns),
                        dtype=np.float32
                    )
                )

        x = np.array(
            x,
            dtype=np.float32
        )

        x = torch.from_numpy(x)

        # -------------------------------------------------
        # Create PyTorch Geometric Data Object
        # -------------------------------------------------
        data = Data(
            x=x,
            edge_index=edge_index
        )

        print(f"Node Features : {data.num_node_features}")
        print(f"Nodes         : {data.num_nodes:,}")
        print(f"Edges         : {data.num_edges:,}")

        return data