import glob
import json

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "Study_Hours_Per_Day",
    "Attendance_Percent",
    "Previous_Semester_Score",
    "Assignments_Completed_Percent",
]
TARGET = "Performance_Class"


def load_dataset():
    # Works whether the file was uploaded as .csv or .xls (it is CSV text either way)
    files = glob.glob("student_performance_classification.*")
    if not files:
        raise FileNotFoundError("student_performance_classification file not found")
    return pd.read_csv(files[0])


def train_model():
    print("Loading dataset...")
    data = load_dataset()
    print("Dataset loaded successfully.")
    print("Number of records:", len(data))

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print("Training records:", len(X_train))
    print("Testing records :", len(X_test))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=["High", "Medium", "Low"])

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))
    print("\nConfusion Matrix (rows/cols = High, Medium, Low):")
    print(matrix)

    joblib.dump(model, "student_performance_model.pkl")
    print("\nModel saved as student_performance_model.pkl")

    metrics = {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test),
    }
    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)
    print("Metrics saved as metrics.json")

    return accuracy


if __name__ == "__main__":
    train_model()
