"""
GraphShield AI
GraphSAGE Trainer
"""

import torch

from src.gnn.graphsage_model import GraphSAGE


class GraphSAGETrainer:

    @staticmethod
    def train(data):

        print("\n" + "=" * 60)
        print("TRAINING GRAPHSAGE MODEL")
        print("=" * 60)

        # -----------------------------
        # Validate Dataset
        # -----------------------------

        if not hasattr(data, "train_mask"):
            raise ValueError("train_mask not found.")

        if not hasattr(data, "y"):
            raise ValueError("Node labels not found.")

        # -----------------------------
        # Create Model
        # -----------------------------

        model = GraphSAGE(
            input_dim=data.num_node_features,
            hidden_dim=64,
            output_dim=2
        )

        # -----------------------------
        # Optimizer
        # -----------------------------

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.005,
            weight_decay=5e-4
        )

        # -----------------------------
        # Class Weights
        # -----------------------------

        train_labels = data.y[data.train_mask]

        num_normal = (train_labels == 0).sum().item()
        num_fraud = (train_labels == 1).sum().item()

        print(f"Training Normal Nodes : {num_normal}")
        print(f"Training Fraud Nodes  : {num_fraud}")

        class_weights = torch.tensor(
            [
                1.0,
                num_normal / max(num_fraud, 1)
            ],
            dtype=torch.float,
            device=data.x.device
        )

        criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights
        )

        # -----------------------------
        # Training Loop
        # -----------------------------

        model.train()

        for epoch in range(1, 101):

            optimizer.zero_grad()

            out = model(data)

            loss = criterion(
                out[data.train_mask],
                data.y[data.train_mask]
            )

            loss.backward()

            optimizer.step()

            if epoch % 10 == 0:

                prediction = out.argmax(dim=1)

                train_accuracy = (
                    prediction[data.train_mask]
                    == data.y[data.train_mask]
                ).float().mean()

                print(
                    f"Epoch {epoch:03d} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Train Accuracy: {train_accuracy:.4f}"
                )

        print("\nGraphSAGE Training Finished")

        return model