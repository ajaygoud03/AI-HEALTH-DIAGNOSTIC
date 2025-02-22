import pandas as pd
import numpy as np
from mat import diseases

# Load dataset
df = pd.read_csv("disease_symptom_dataset.csv")  # Make sure the path is correct

# Extract disease column
diseases = df["disease"]  # Make sure the column name is correct

# Count unique disease occurrences
unique, counts = np.unique(diseases, return_counts=True)

# Print results
print(dict(zip(unique, counts)))


