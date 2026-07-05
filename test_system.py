"""
Focused tests for the consolidated attendance app.
"""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from config import Config
from db import DatabaseManager
from main import AttendanceSystem
from recognition import FaceRecognitionSystem, detect_faces, face_backend_available, recognize_and_log
from setup_app import FULL_VENV, MINIMAL_VENV, parse_args, venv_python


class TempDatabaseTest(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.db = DatabaseManager(self.db_path)

    def tearDown(self):
        try:
            os.remove(self.db_path)
        except FileNotFoundError:
            pass


class TestConfig(unittest.TestCase):
    def test_validation(self):
        self.assertTrue(Config.validate_tolerance(0.6))
        self.assertFalse(Config.validate_tolerance(0.9))
        self.assertTrue(Config.validate_confidence(0.5))
        self.assertFalse(Config.validate_confidence(1.5))
        self.assertTrue(Config.validate_camera_index(0))
        self.assertFalse(Config.validate_camera_index(-1))

    def test_dependency_report_shape(self):
        report = Config.dependency_report(check_camera=False)
        self.assertIn("minimal_ready", report)
        self.assertIn("full_ready", report)
        self.assertIn("missing_minimal", report)


class TestBootstrapper(unittest.TestCase):
    def test_default_bootstrap_profile_is_minimal(self):
        args = parse_args([])
        self.assertEqual(args.profile, "minimal")
        self.assertFalse(args.run)
        self.assertFalse(args.check_only)

    def test_full_profile_uses_separate_venv(self):
        args = parse_args(["--profile", "full", "--check-only"])
        self.assertEqual(args.profile, "full")
        self.assertTrue(args.check_only)
        self.assertIn(".venv-full", str(venv_python(FULL_VENV)))

    def test_minimal_profile_uses_default_venv(self):
        self.assertIn(".venv", str(venv_python(MINIMAL_VENV)))


class TestDatabaseManager(TempDatabaseTest):
    def test_student_embedding_and_statistics(self):
        student_id = self.db.add_student("Test Student", "TEST001", category="Class 1A", image_paths=["face.jpg"])
        self.assertEqual(self.db.get_student_by_id(student_id)[1], "Test Student")
        self.assertTrue(self.db.add_face_embedding(student_id, [0.1] * 128))
        self.assertEqual(len(self.db.get_face_embeddings(student_id)), 1)
        people = self.db.list_registered_people()
        self.assertEqual(people[0]["category"], "Class 1A")
        self.assertEqual(people[0]["image_count"], 1)

        stats = self.db.get_statistics()
        self.assertEqual(stats["total_students"], 1)
        self.assertEqual(stats["total_embeddings"], 1)

    def test_duplicate_enrollment_validation(self):
        self.db.add_student("Student 1", "DUP001")
        with self.assertRaises(ValueError):
            self.db.add_student("Student 2", "DUP001")

    def test_duplicate_attendance_suppression(self):
        student_id = self.db.add_student("Test Student", "TEST002")
        first = self.db.log_attendance(student_id, "morning", 0.9)
        second = self.db.log_attendance(student_id, "morning", 0.9)
        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(len(self.db.get_attendance_logs()), 1)

    def test_duplicate_attendance_suppression_is_session_based(self):
        student_id = self.db.add_student("Test Student", "TEST003")
        with self.db._connect() as conn:
            conn.execute(
                "INSERT INTO attendance_logs (student_id, session, confidence, timestamp) VALUES (?, ?, ?, ?)",
                (student_id, "morning", 0.9, "2000-01-01 00:00:00"),
            )
            conn.commit()

        suppressed = self.db.log_attendance(student_id, "morning", 0.9, duplicate_window_seconds=1)

        self.assertFalse(suppressed)
        self.assertEqual(len(self.db.get_attendance_logs()), 1)

    def test_csv_export_even_when_empty(self):
        fd, csv_path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            self.assertTrue(self.db.export_to_csv(csv_path))
            with open(csv_path, "r", encoding="utf-8") as exported:
                self.assertIn("Student ID", exported.read())
        finally:
            os.remove(csv_path)

    def test_delete_student_removes_related_records(self):
        student_id = self.db.add_student("Delete Me", "DEL001", category="Staff")
        self.db.add_face_embedding(student_id, [0.1] * 128)
        self.db.log_attendance(student_id, "morning", 0.9)
        self.assertTrue(self.db.delete_student(student_id))
        self.assertEqual(self.db.get_student_by_id(student_id), None)
        self.assertEqual(self.db.get_face_embeddings(student_id), [])
        self.assertEqual(self.db.get_attendance_logs(), [])


class TestAttendanceSystem(TempDatabaseTest):
    def test_manual_enrollment_and_attendance_without_face_backend(self):
        system = AttendanceSystem(self.db_path)
        result = system.register_student(
            "Manual Student",
            enrollment_number="MAN001",
            category="Class 2B",
            image_paths=["sample.jpg"],
            allow_manual_without_face=True,
        )
        self.assertTrue(result["success"])
        self.assertTrue(result["manual_only"])
        self.assertEqual(result["category"], "Class 2B")

        log_result = system.log_manual_attendance("MAN001", "class-a")
        self.assertTrue(log_result["success"])
        self.assertEqual(len(system.get_attendance_logs()), 1)

    def test_delete_registered_person_reloads_known_students(self):
        system = AttendanceSystem(self.db_path)
        result = system.register_student("Temp Person", enrollment_number="TMP001", allow_manual_without_face=True)
        self.assertTrue(result["success"])
        self.assertEqual(len(system.list_registered_people()), 1)
        self.assertTrue(system.delete_registered_person(result["student_id"]))
        self.assertEqual(system.list_registered_people(), [])

    def test_known_students_cache(self):
        system = AttendanceSystem(self.db_path)
        student_id = system.db_manager.add_student("Known Student", "KNOWN001")
        system.db_manager.add_face_embedding(student_id, [0.2] * 128)
        system.load_known_students()
        if face_backend_available():
            self.assertEqual(len(system.face_system.known_encodings), 1)
        else:
            self.assertEqual(system.known_students[0][1], "Known Student")


class TestRecognition(unittest.TestCase):
    def test_detect_faces_is_safe_without_backend(self):
        self.assertIsInstance(detect_faces(None), list)

    def test_recognize_and_log_uses_cached_known_data(self):
        frame = Mock()
        frame.copy.return_value = frame
        db = Mock()
        with patch("recognition.face_backend_available", return_value=False):
            self.assertIs(recognize_and_log(frame, [], db), frame)
            db.log_attendance.assert_not_called()

    def test_face_system_validate_embedding(self):
        system = FaceRecognitionSystem()
        self.assertTrue(system.validate_embedding([0.0] * 128))
        self.assertFalse(system.validate_embedding([0.0] * 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
