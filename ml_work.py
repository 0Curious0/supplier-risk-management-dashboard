import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from supplier_data import df


# Select relevant features
features = ["Supplier_ID", "Delay_Days", "Supplier_Reliability_Score", "Parameter_Change_Magnitude"]
target = "Supply_Risk_Flag"

# Prepare feature matrix (X) and target (y)
X = df[["Delay_Days", "Supplier_Reliability_Score", "Parameter_Change_Magnitude"]]
y = df[target]    

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = GaussianNB()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

df["Predicted_Prob_Risk"] = model.predict_proba(scaler.transform(X))[:, 1]

