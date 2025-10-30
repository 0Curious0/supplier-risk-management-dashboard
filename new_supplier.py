import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ml_work import model

# Example: data for a new supplier
new_supplier_data = {
    "Delay_Days": 10,
    "Supplier_Reliability_Score": 50,
    "Parameter_Change_Magnitude": 2.5
}

scaler = StandardScaler()

# Convert to DataFrame to match training structure
new_supplier_df = pd.DataFrame([new_supplier_data])

# Apply the same scaling used in training
new_supplier_scaled = scaler.fit_transform(new_supplier_df)

# Predict the supply risk
predicted_class = model.predict(new_supplier_scaled)
predicted_prob = model.predict_proba(new_supplier_scaled)

# Interpret results
if predicted_class[0] == 1:
    print("⚠️ High Supply Risk supplier detected!")
else:
    print("✅ Supplier seems reliable.")

print(f"Probability of being risky: {predicted_prob[0][1]*100:.2f}%")
print(f"Probability of being safe: {predicted_prob[0][0]*100:.2f}%")