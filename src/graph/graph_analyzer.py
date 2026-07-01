"""
GraphShield AI
Graph Analytics Module
"""

import networkx as nx


class GraphAnalyzer:

    @staticmethod
    def analyze(G):

        print("\n" + "=" * 60)
        print("GRAPH ANALYTICS")
        print("=" * 60)

        print(f"Nodes              : {G.number_of_nodes():,}")
        print(f"Edges              : {G.number_of_edges():,}")

        print(f"Density            : {nx.density(G):.8f}")

        avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()

        print(f"Average Degree     : {avg_degree:.2f}")

        print(f"Weak Components    : {nx.number_weakly_connected_components(G)}")

        print("\nTop 10 Customers by Degree")

        degree = sorted(
            G.degree(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        for node, deg in degree:
            print(f"{node:<15} {deg}")