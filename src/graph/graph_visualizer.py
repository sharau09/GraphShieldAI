"""
GraphShield AI
Graph Visualization
"""

from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx


class GraphVisualizer:

    @staticmethod
    def visualize(G):

        print("\nGenerating Network Visualization...")

        output_dir = Path("outputs/graphs")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Small subgraph for visualization
        nodes = list(G.nodes())[:150]

        subgraph = G.subgraph(nodes)

        plt.figure(figsize=(12, 10))

        pos = nx.spring_layout(
            subgraph,
            seed=42
        )

        nx.draw_networkx_nodes(
            subgraph,
            pos,
            node_size=40
        )

        nx.draw_networkx_edges(
            subgraph,
            pos,
            alpha=0.5,
            arrows=False
        )

        plt.title("Customer Transaction Network")

        plt.axis("off")

        plt.tight_layout()

        plt.savefig(
            output_dir / "customer_network.png",
            dpi=300
        )

        plt.close()

        print("✓ customer_network.png created")