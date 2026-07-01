"""
GraphShield AI
Customer Transaction Graph Builder
"""

import networkx as nx


class GraphBuilder:

    @staticmethod
    def build_graph(df):

        print("\nBuilding Customer Transaction Graph...")

        G = nx.DiGraph()

        for _, row in df.iterrows():

            G.add_edge(
                row["nameOrig"],
                row["nameDest"],
                amount=row["amount"],
                transaction_type=row["type"],
                fraud=row["isFraud"],
                step=row["step"]
            )

        print(f"Nodes : {G.number_of_nodes():,}")
        print(f"Edges : {G.number_of_edges():,}")

        return G