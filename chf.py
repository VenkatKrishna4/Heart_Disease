import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 👉 Load dataset
df = pd.read_csv("heart.csv")
print(df.head())

# 👉 Check for missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# 👉 Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns

# 👉 Encode categorical columns
encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

print("\nCategorical columns encoded successfully!")

# 👉 Handle missing values (only for numeric columns)
df.fillna(df.median(numeric_only=True), inplace=True)

print("\nMissing values filled with median values!")

# 👉 Define features (X) and target (y)
X = df.drop(columns=["HeartDisease"])  # Assuming "HeartDisease" is the target column
y = df["HeartDisease"]

# 👉 Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining Set: {X_train.shape}, Test Set: {X_test.shape}")

# 👉 Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\nModel training completed!")

# 👉 Evaluate model performance
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 👉 Save the trained model
with open("chf_prediction_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\n✅ Model saved as 'chf_prediction_model.pkl'")
