"""
GraphShield AI
Train / Validation / Test Masks
"""

import torch


class GraphMasks:

    @staticmethod
    def create(data):

        print("\n" + "=" * 60)
        print("CREATING TRAIN / VALIDATION / TEST MASKS")
        print("=" * 60)

        num_nodes = data.num_nodes

        torch.manual_seed(42)

        indices = torch.randperm(num_nodes)

        train_size = int(0.70 * num_nodes)
        val_size = int(0.15 * num_nodes)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[indices[:train_size]] = True

        val_mask[
            indices[
                train_size:
                train_size + val_size
            ]
        ] = True

        test_mask[
            indices[
                train_size + val_size:
            ]
        ] = True

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        print(f"Training Nodes   : {train_mask.sum().item():,}")
        print(f"Validation Nodes : {val_mask.sum().item():,}")
        print(f"Testing Nodes    : {test_mask.sum().item():,}")

        return data