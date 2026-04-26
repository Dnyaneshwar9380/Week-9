import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("customer_churn.csv")

# Fill missing values
df.fillna(method='ffill', inplace=True)

# Encode categorical columns
le_dict = {}

for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Split data
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model + columns
joblib.dump(model, "churn_model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("Model Saved Successfully!")