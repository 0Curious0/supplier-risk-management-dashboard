import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Load data
df = pd.read_csv("data.csv")

df["Delay_Days"] = df["Delay_Days"] + np.random.normal(0, 0.5, size=len(df))
df["Delay_Days"] = df["Delay_Days"].clip(lower=0)

mask = (df["Delay_Days"] < 2) & (df["Supply_Risk_Flag"] == 1)
df.loc[mask, "Supply_Risk_Flag"] = np.random.choice([0, 1], size=mask.sum())




