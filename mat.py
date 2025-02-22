import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# Load dataset (modify the path if needed)
df = pd.read_csv("disease_symptom_dataset.csv")

# Ensure 'disease' column exists before dropping
if "disease" in df.columns:
    symptoms = df.drop(columns=["disease"])
else:
    print("Warning: 'disease' column not found!")
    symptoms = df  # Fallback

# Handle missing values and convert all to strings
symptoms = symptoms.fillna("").astype(str)

# Convert each row into a list of symptoms
symptoms_list = symptoms.apply(lambda row: row.tolist(), axis=1)

# Initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer()

# Fit and transform symptoms
symptoms_encoded = mlb.fit_transform(symptoms_list)
symptoms_list = symptoms.apply(lambda row: [symptom for symptom in row if symptom.strip()], axis=1)
symptoms_encoded = mlb.fit_transform(symptoms_list)
mlb.classes_ = [s.strip() for s in mlb.classes_]

# Debugging output
print("Sample Encoded Symptoms:", symptoms_encoded[0])
print("Encoded Symptoms Shape:", symptoms_encoded.shape)
print("Classes:", mlb.classes_)
# mat.py


# Load the dataset
df = pd.read_csv("disease_symptom_dataset.csv")  # Ensure correct file path

# Extract diseases column
diseases = df["disease"].tolist()  # Convert to list if needed
