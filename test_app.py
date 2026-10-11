import unittest

from app import app


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_high_performance_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "Study_Hours_Per_Day": 5,
                "Attendance_Percent": 95,
                "Previous_Semester_Score": 90,
                "Assignments_Completed_Percent": 95
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["prediction"],
            "High"
        )

    def test_low_performance_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "Study_Hours_Per_Day": 1,
                "Attendance_Percent": 60,
                "Previous_Semester_Score": 35,
                "Assignments_Completed_Percent": 40
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["prediction"],
            "High"
        )

    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict",
            json={
                "Study_Hours_Per_Day": 5,
                "Attendance_Percent": 95
            }
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "missing_fields",
            response.get_json()
        )


if __name__ == "__main__":
    unittest.main()
