import pandas as pd

df = pd.read_csv("healthcare-dataset-stroke-data.csv")

print(df.head())


print("Missing Values in Each Column: ")
print(df.isnull().sum())

df["bmi"].fillna(df["bmi"].median(),inplace=True)
print("Missing BMI Values filled!")

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df["gender"] = encoder.fit_transform(df["gender"])
df["ever_married"] = encoder.fit_transform(df["ever_married"])
df["work_type"] = encoder.fit_transform(df["work_type"])
df["Residence_type"] = encoder.fit_transform(df["Residence_type"])
df["smoking_status"] = encoder.fit_transform(df["smoking_status"])

print("Categorical columns encoded sucessfully")


df.drop(columns=["id"],inplace=True)
print("ID column removed!")



from sklearn.model_selection import train_test_split

x = df.drop(columns=["stroke"])
y = df["stroke"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)


print(f"Training Set: {x_train.shape}, Test Set:{x_test.shape}")


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
print("Model training completed!")

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(x_test)

print(f"Model Accuracy: {accuracy_score(y_test,y_pred)*100: .2f}%")
print("\n Classification Report: \n",classification_report(y_test,y_pred))


import pickle

# Save the trained model
with open("stroke_prediction_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved as 'stroke_prediction_model.pkl")
