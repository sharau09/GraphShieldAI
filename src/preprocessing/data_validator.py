"""
GraphShield AI
Dataset Validation Module
"""


class DataValidator:

    REQUIRED_COLUMNS = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud",
    ]

    @classmethod
    def validate(cls, df):

        print("=" * 60)
        print("Validating Dataset...")
        print("=" * 60)

        missing = [
            col
            for col in cls.REQUIRED_COLUMNS
            if col not in df.columns
        ]

        if missing:
            raise Exception(
                f"Missing Columns: {missing}"
            )

        print("All required columns are present.")
        print(f"Rows    : {len(df):,}")
        print(f"Columns : {len(df.columns)}")