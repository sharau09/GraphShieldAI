"""
GraphShield AI
PyTorch Geometric Dataset Builder
"""

import torch
from torch_geometric.data import Data


class GNNDataset:

    @staticmethod
    def build(graph):

        print("\n" + "=" * 60)
        print("BUILDING PYTORCH GEOMETRIC DATASET")
        print("=" * 60)

        node_mapping = {
            node: i
            for i, node in enumerate(graph.nodes())
        }

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

        x = torch.ones(
            (len(node_mapping), 1),
            dtype=torch.float
        )

        data = Data(
            x=x,
            edge_index=edge_index
        )

        print(f"Nodes : {data.num_nodes:,}")
        print(f"Edges : {data.num_edges:,}")

        return data