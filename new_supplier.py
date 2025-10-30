import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ml_work import model, scaler

# Example: data for a new supplier
def new_sup(delay, score, parameter):
    new_supplier_data = {
        "Delay_Days": delay,
        "Supplier_Reliability_Score": score,
        "Parameter_Change_Magnitude": parameter
    }


    # Convert to DataFrame to match training structure
    new_supplier_df = pd.DataFrame([new_supplier_data])

    # Apply the same scaling used in training
    new_supplier_scaled = scaler.transform(new_supplier_df)

    # Predict the supply risk
    predicted_class = model.predict(new_supplier_scaled)
    predicted_prob = model.predict_proba(new_supplier_scaled)

    # Interpret results
    if predicted_class[0] == 1:
        print("⚠️ High Supply Risk supplier detected!")
        risk_level = "🔴 High Risk"
        color = "#e74c3c"
        show_balloons = False
        
    else:
        print("✅ Supplier seems reliable.")
        risk_level = "🟢 Low Risk"
        color = "#27ae60"
        show_balloons = True

    print(f"Probability of being risky: {predicted_prob[0][1]*100:.2f}%")
    print(f"Probability of being safe: {predicted_prob[0][0]*100:.2f}%")

    return risk_level, color, show_balloons