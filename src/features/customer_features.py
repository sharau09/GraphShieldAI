"""
GraphShield AI
Customer Feature Engineering
"""

import pandas as pd


class CustomerFeatures:

    @staticmethod
    def generate(transactions, graph_features, risk_scores):

        print("\n" + "=" * 60)
        print("CUSTOMER FEATURE ENGINEERING")
        print("=" * 60)

        # -------------------------------------------------
        # Customer Behaviour Features
        # -------------------------------------------------

        customer_features = transactions.groupby("nameOrig").agg(

            transaction_count=("amount", "count"),

            total_sent_amount=("amount", "sum"),

            average_transaction_amount=("amount", "mean"),

            max_transaction_amount=("amount", "max"),

            cash_out_count=(
                "type",
                lambda x: (x == "CASH_OUT").sum()
            ),

            transfer_count=(
                "type",
                lambda x: (x == "TRANSFER").sum()
            ),

            payment_count=(
                "type",
                lambda x: (x == "PAYMENT").sum()
            ),

            fraud_ratio=(
                "isFraud",
                "mean"
            )

        ).reset_index()

        customer_features.rename(
            columns={
                "nameOrig": "customer"
            },
            inplace=True
        )

        # -------------------------------------------------
        # Merge Graph Features
        # -------------------------------------------------

        customer_features = customer_features.merge(
            graph_features,
            on="customer",
            how="left"
        )

        # -------------------------------------------------
        # Merge Risk Scores
        # -------------------------------------------------

        customer_features = customer_features.merge(
            risk_scores,
            on="customer",
            how="left"
        )

        # -------------------------------------------------
        # Replace Missing Values
        # -------------------------------------------------

        customer_features.fillna(0, inplace=True)

        # -------------------------------------------------
        # Debug Output
        # -------------------------------------------------

        print(f"Customers : {len(customer_features):,}")

        print("\nColumns\n")
        print(customer_features.columns.tolist())

        print("\nTop 5 Records\n")
        print(customer_features.head())

        print("\n==============================")
        print("CUSTOMER FEATURE COLUMNS")
        print("==============================")
        print(customer_features.columns.tolist())

        return customer_features