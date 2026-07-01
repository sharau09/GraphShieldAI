"""
GraphShield AI
Data Profiling Module
"""

from pathlib import Path


class DataProfiler:

    @staticmethod
    def generate_report(df):

        print("\n" + "=" * 60)
        print("DATASET PROFILE")
        print("=" * 60)

        print(f"Rows               : {df.shape[0]:,}")
        print(f"Columns            : {df.shape[1]}")
        print(f"Duplicate Rows     : {df.duplicated().sum():,}")

        print("\nMissing Values")
        print("-" * 60)
        print(df.isnull().sum())

        print("\nData Types")
        print("-" * 60)
        print(df.dtypes)

        print("\nFraud Distribution")
        print("-" * 60)
        print(df["isFraud"].value_counts())

        print("\nTransaction Types")
        print("-" * 60)
        print(df["type"].value_counts())

        report_path = Path("outputs/reports")
        report_path.mkdir(parents=True, exist_ok=True)

        with open(report_path / "dataset_report.txt", "w") as f:
            f.write("GraphShield AI Dataset Report\n")
            f.write("=" * 40 + "\n")
            f.write(f"Rows: {df.shape[0]}\n")
            f.write(f"Columns: {df.shape[1]}\n")
            f.write(f"Duplicate Rows: {df.duplicated().sum()}\n")

        print("\nReport saved to outputs/reports/dataset_report.txt")