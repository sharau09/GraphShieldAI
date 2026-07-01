from src.preprocessing.data_loader import DataLoader
from src.preprocessing.data_validator import DataValidator
from src.preprocessing.data_profiler import DataProfiler
from src.preprocessing.sample_generator import SampleGenerator

from src.analysis.eda import EDA

from src.graph.graph_builder import GraphBuilder
from src.graph.graph_analyzer import GraphAnalyzer
from src.graph.graph_visualizer import GraphVisualizer

from src.fraud_ring.fraud_ring_detector import FraudRingDetector
from src.fraud_ring.risk_score import RiskScore

from src.features.graph_features import GraphFeatures
from src.features.customer_features import CustomerFeatures

from src.models.dataset_builder import DatasetBuilder
from src.models.baseline_model import BaselineModel
from src.models.xgboost_model import XGBoostModel
from src.models.compare_models import ModelComparison

from src.gnn.feature_dataset import GNNFeatureDataset
from src.gnn.label_builder import LabelBuilder
from src.gnn.masks import GraphMasks

from src.gnn.train import GCNTrainer
from src.gnn.evaluate import GCNEvaluator

from src.gnn.graphsage_train import GraphSAGETrainer
from src.gnn.graphsage_evaluate import GraphSAGEEvaluator

from src.explainability.shap_explainer import SHAPExplainer


def main():

    # ==========================================================
    # Load Dataset
    # ==========================================================

    loader = DataLoader()
    df = loader.load_dataset()

    # ==========================================================
    # Validate Dataset
    # ==========================================================

    DataValidator.validate(df)

    # ==========================================================
    # Dataset Profile
    # ==========================================================

    DataProfiler.generate_report(df)

    # ==========================================================
    # Create Development Sample
    # ==========================================================

    sample_df = SampleGenerator.create_sample(df)

    # ==========================================================
    # Exploratory Data Analysis
    # ==========================================================

    EDA.fraud_distribution(sample_df)
    EDA.transaction_type_distribution(sample_df)

    # ==========================================================
    # Build Customer Transaction Graph
    # ==========================================================

    graph = GraphBuilder.build_graph(sample_df)

    # ==========================================================
    # Graph Analytics
    # ==========================================================

    GraphAnalyzer.analyze(graph)

    # ==========================================================
    # Graph Visualization
    # ==========================================================

    GraphVisualizer.visualize(graph)

    # ==========================================================
    # Fraud Ring Detection
    # ==========================================================

    FraudRingDetector.detect(graph)

    # ==========================================================
    # Graph Feature Engineering
    # ==========================================================

    graph_features = GraphFeatures.generate(graph)

    # ==========================================================
    # Customer Risk Scores
    # ==========================================================

    risk_scores = RiskScore.generate(graph_features)

    # ==========================================================
    # Customer Feature Engineering
    # ==========================================================

    customer_features = CustomerFeatures.generate(
        sample_df,
        graph_features,
        risk_scores
    )

    # ==========================================================
    # Build ML Training Dataset
    # ==========================================================

    training_data = DatasetBuilder.build(
        sample_df,
        graph_features
    )

    # ==========================================================
    # Train Traditional ML Models
    # ==========================================================

    print("\nTraining Traditional ML Models...\n")

    rf = BaselineModel.train(training_data)
    xgb = XGBoostModel.train(training_data)

    rf_results = rf["metrics"]
    xgb_results = xgb["metrics"]

    # ==========================================================
    # Build GNN Dataset
    # ==========================================================

    gnn_data = GNNFeatureDataset.build(
        graph,
        customer_features
    )

    # ==========================================================
    # Build Node Labels
    # ==========================================================

    gnn_data.y = LabelBuilder.build(
        graph,
        sample_df
    )

    # ==========================================================
    # Create Train / Validation / Test Masks
    # ==========================================================

    gnn_data = GraphMasks.create(gnn_data)

    # ==========================================================
    # Train GCN
    # ==========================================================

    gcn_model = GCNTrainer.train(
        gnn_data
    )

    # ==========================================================
    # Evaluate GCN
    # ==========================================================

    gcn_results = GCNEvaluator.evaluate(
        gcn_model,
        gnn_data
    )

    # ==========================================================
    # Train GraphSAGE
    # ==========================================================

    graphsage_model = GraphSAGETrainer.train(
        gnn_data
    )

    # ==========================================================
    # Evaluate GraphSAGE
    # ==========================================================

    graphsage_results = GraphSAGEEvaluator.evaluate(
        graphsage_model,
        gnn_data
    )

    # ==========================================================
    # SHAP Explainability
    # ==========================================================

    print("\nGenerating SHAP Explainability...\n")

    SHAPExplainer.explain(
    xgb["model"],
    xgb["X_test"]
)

    # ==========================================================
    # Compare Models
    # ==========================================================

    ModelComparison.save([
        rf_results,
        xgb_results,
        gcn_results,
        graphsage_results
    ])

    # ==========================================================
    # Pipeline Completed
    # ==========================================================

    print("\n" + "=" * 60)
    print("GRAPHSHIELD AI PIPELINE COMPLETED")
    print("=" * 60)

    print(f"Total Nodes : {graph.number_of_nodes():,}")
    print(f"Total Edges : {graph.number_of_edges():,}")


if __name__ == "__main__":
    main()