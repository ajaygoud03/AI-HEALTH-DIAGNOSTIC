
import pandas as pd  

# Load your dataset (modify the path as needed)
df = pd.read_csv("disease_symptom_dataset.csv")  

# Ensure 'disease' column exists before dropping
if "disease" in df.columns:
    symptoms = df.drop(columns=["disease"])
else:
    print("Warning: 'disease' column not found!")
    symptoms = df  # Fallback
symptoms = symptoms.fillna("")
symptoms = symptoms.astype(str)
symptoms_list = symptoms.apply(lambda row: row.tolist(), axis=1)

