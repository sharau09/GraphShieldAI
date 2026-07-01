"""
GraphShield AI
Graph Feature Engineering
"""

from pathlib import Path
import pandas as pd
import networkx as nx


class GraphFeatures:

    @staticmethod
    def generate(G):

        print("\n" + "=" * 60)
        print("GRAPH FEATURE ENGINEERING")
        print("=" * 60)

        print("Calculating Degree...")
        degree = dict(G.degree())

        print("Calculating In Degree...")
        in_degree = dict(G.in_degree())

        print("Calculating Out Degree...")
        out_degree = dict(G.out_degree())

        print("Calculating PageRank...")
        pagerank = nx.pagerank(G)

        print("Calculating Betweenness...")
        betweenness = nx.betweenness_centrality(
            G,
            k=100,
            seed=42
        )

        print("Calculating Clustering Coefficient...")
        clustering = nx.clustering(G.to_undirected())

        data = []

        for node in G.nodes():

            data.append({
                "customer": node,
                "degree": degree.get(node, 0),
                "in_degree": in_degree.get(node, 0),
                "out_degree": out_degree.get(node, 0),
                "pagerank": pagerank.get(node, 0),
                "betweenness": betweenness.get(node, 0),
                "clustering": clustering.get(node, 0)
            })

        features = pd.DataFrame(data)

        output = Path("outputs/reports")
        output.mkdir(parents=True, exist_ok=True)

        features.to_csv(
            output / "graph_features.csv",
            index=False
        )

        print(f"\nGenerated Features : {len(features):,}")

        print("\nTop 10 Records\n")

        print(features.head(10))

        return features