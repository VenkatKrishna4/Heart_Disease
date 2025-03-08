import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("heart.csv")
print(df.head())

print("\nMissing Values in Each Column:")
print(df.isnull().sum())

categorical_columns = df.select_dtypes(include=['object']).columns

encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

print("\nCategorical columns encoded successfully!")

df.fillna(df.median(numeric_only=True), inplace=True)

print("\nMissing values filled with median values!")

X = df.drop(columns=["HeartDisease"])  # Assuming "HeartDisease" is the target column
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining Set: {X_train.shape}, Test Set: {X_test.shape}")

#model for dataset
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\nModel training completed!")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

with open("chf_prediction_model.pkl", "wb") as file:
    pickle.dump(model, file)

