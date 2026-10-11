import glob
import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_dataset_exists(self):
        self.assertTrue(len(glob.glob("student_performance_classification.*")) > 0)

    def test_model_created(self):
        self.assertTrue(os.path.exists("student_performance_model.pkl"))

    def test_metrics_created(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)
        accuracy = metrics["accuracy"]
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_prediction(self):
        model = joblib.load("student_performance_model.pkl")
        sample = pd.DataFrame([{
            "Study_Hours_Per_Day": 3,
            "Attendance_Percent": 85,
            "Previous_Semester_Score": 65,
            "Assignments_Completed_Percent": 75,
        }])
        prediction = model.predict(sample)[0]
        self.assertIn(prediction, ["High", "Medium", "Low"])

    def test_high_performance_student(self):
        model = joblib.load("student_performance_model.pkl")
        sample = pd.DataFrame([{
            "Study_Hours_Per_Day": 5,
            "Attendance_Percent": 95,
            "Previous_Semester_Score": 90,
            "Assignments_Completed_Percent": 95,
        }])
        prediction = model.predict(sample)[0]
        self.assertEqual(prediction, "High")

    def test_low_performance_student(self):
        model = joblib.load("student_performance_model.pkl")
        sample = pd.DataFrame([{
            "Study_Hours_Per_Day": 1,
            "Attendance_Percent": 60,
            "Previous_Semester_Score": 35,
            "Assignments_Completed_Percent": 40,
        }])
        prediction = model.predict(sample)[0]
        self.assertEqual(prediction, "Low")


if __name__ == "__main__":
    unittest.main()
