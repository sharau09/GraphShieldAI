"""
GraphShield AI
GCN Trainer
"""

import torch

from src.gnn.gcn_model import GCN


class GCNTrainer:

    @staticmethod
    def train(data):

        print("\n" + "=" * 60)
        print("TRAINING GRAPH CONVOLUTIONAL NETWORK")
        print("=" * 60)

        if not hasattr(data, "train_mask"):
            raise ValueError("train_mask not found.")

        if not hasattr(data, "y"):
            raise ValueError("Labels not found.")

        model = GCN(
            input_dim=data.num_node_features,
            hidden_dim=32,
            output_dim=2
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.01,
            weight_decay=5e-4
        )

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

        model.train()

        for epoch in range(1, 51):

            optimizer.zero_grad()

            out = model(data)

            loss = criterion(
                out[data.train_mask],
                data.y[data.train_mask]
            )

            loss.backward()

            optimizer.step()

            if epoch % 5 == 0:

                prediction = out.argmax(dim=1)

                train_accuracy = (
                    prediction[data.train_mask]
                    == data.y[data.train_mask]
                ).float().mean()

                print(
                    f"Epoch {epoch:02d} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Train Accuracy: {train_accuracy:.4f}"
                )

        print("\nGCN Training Finished")

        return model