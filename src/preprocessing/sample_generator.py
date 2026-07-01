"""
GraphShield AI
Sample Generator
"""

from pathlib import Path


class SampleGenerator:

    @staticmethod
    def create_sample(df, sample_size=100000):

        print("\nCreating development dataset...")

        sample = df.sample(
            n=sample_size,
            random_state=42
        )

        output_path = Path("data/processed/paysim_sample_100k.csv")

        sample.to_csv(output_path, index=False)

        print(f"Sample saved to:\n{output_path}")

        return sample