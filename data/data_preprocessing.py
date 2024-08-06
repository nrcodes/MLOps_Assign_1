import pandas as pd

# Load the dataset
df = pd.read_csv("breast_cancer_dataset.csv")

# Add a new feature (e.g., square of the mean radius)
df["mean radius squared modified"] = df["mean radius"] ** 2

# Save the modified dataset
df.to_csv("breast_cancer_modified.csv", index=False)
