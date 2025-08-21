# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load dataset
df = pd.read_csv("synthetic_job_data.csv")

# Encode categorical values
le_edu = LabelEncoder()
le_applied = LabelEncoder()
le_cover = LabelEncoder()

df["Education"] = le_edu.fit_transform(df["Education"])
df["AppliedBefore"] = le_applied.fit_transform(df["AppliedBefore"])
df["CoverLetter"] = le_cover.fit_transform(df["CoverLetter"])

# Split features and target
X = df.drop("Shortlisted", axis=1)
y = df["Shortlisted"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model and encoders
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/job_success_model.pkl")
joblib.dump(le_edu, "model/le_edu.pkl")
joblib.dump(le_applied, "model/le_applied.pkl")
joblib.dump(le_cover, "model/le_cover.pkl")

print("✅ Model and encoders saved in 'model' folder")
