"""
Entry point for the local face attendance system.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import Config, ErrorHandler
from db import DatabaseManager
from recognition import (
    FaceRecognitionSystem,
    capture_images,
    computer_vision_available,
    detect_faces,
    extract_embeddings,
    face_backend_available,
    get_system_status as get_recognition_status,
    recognize_and_log,
)

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

Config.ensure_directories()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(Config.LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class AttendanceSystem:
    """Coordinates database, recognition, GUI, and CLI workflows."""

    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path or Config.DATABASE_PATH
        self.db_manager: Optional[DatabaseManager] = None
        self.face_system: Optional[FaceRecognitionSystem] = None
        self.known_students: List[Any] = []
        self.is_initialized = False
        self._initialize_components()

    def _initialize_components(self) -> None:
        self.db_manager = DatabaseManager(self.database_path)
        self.face_system = FaceRecognitionSystem(Config.RECOGNITION_TOLERANCE, Config.CAMERA_INDEX)
        self.load_known_students()
        self.is_initialized = True

    def load_known_students(self) -> None:
        if not self.db_manager or not self.face_system:
            self.known_students = []
            return
        self.known_students = self.db_manager.get_all_students()
        self.face_system.update_known_students(self.known_students)

    def get_system_status(self) -> Dict[str, Any]:
        dependency_report = Config.dependency_report(check_camera=False)
        recognition_status = get_recognition_status(check_camera=False)
        return {
            "initialized": self.is_initialized,
            "database_connected": self.db_manager is not None,
            "face_recognition_ready": face_backend_available(),
            "minimal_ready": dependency_report["minimal_ready"],
            "full_ready": dependency_report["full_ready"],
            "known_students": len(self.known_students),
            "database_path": self.database_path,
            "default_session": Config.DEFAULT_SESSION,
            "tolerance": self.face_system.tolerance if self.face_system else Config.RECOGNITION_TOLERANCE,
            "database_stats": self.db_manager.get_statistics() if self.db_manager else {},
            "dependencies": dependency_report,
            "recognition": recognition_status,
        }

    def register_student(
        self,
        name: str,
        enrollment_number: Optional[str] = None,
        category: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        num_images: int = 5,
        allow_manual_without_face: bool = False,
    ) -> Dict[str, Any]:
        if not self.is_initialized or not self.db_manager:
            return {"success": False, "error": "System not initialized"}
        if isinstance(enrollment_number, int):
            num_images = enrollment_number
            enrollment_number = None
        try:
            if not Config.validate_image_count(num_images):
                return {"success": False, "error": f"Image count must be between {Config.IMAGE_COUNT_RANGE[0]} and {Config.IMAGE_COUNT_RANGE[1]}"}

            enrollment_number = enrollment_number or f"STU{int(time.time())}"
            embeddings: List[Any] = []

            if image_paths:
                if cv2 is not None and face_backend_available():
                    for image_path in image_paths:
                        image = cv2.imread(image_path)
                        if image is None:
                            logger.warning("Could not read image: %s", image_path)
                            continue
                        face_locations = detect_faces(image)
                        encodings = extract_embeddings(image, face_locations)
                        if encodings:
                            embeddings.append(encodings[0])
                else:
                    logger.info("Stored selected photos without embeddings because automatic recognition is unavailable")

            if not embeddings and face_backend_available():
                embeddings = capture_images(name, num_images, Config.CAMERA_INDEX)

            if not embeddings and not allow_manual_without_face:
                return {"success": False, "error": "No face embeddings captured. Install full dependencies or use manual enrollment."}

            student_id = self.db_manager.add_student(name, enrollment_number, category=category, image_paths=image_paths)
            for embedding in embeddings:
                self.db_manager.add_face_embedding(student_id, embedding)
            self.load_known_students()
            return {
                "success": True,
                "student_id": student_id,
                "name": name,
                "enrollment_number": enrollment_number,
                "category": category or "",
                "embeddings_created": len(embeddings),
                "manual_only": len(embeddings) == 0,
            }
        except Exception as exc:
            logger.error("Registration failed: %s", exc)
            return {"success": False, "error": str(exc)}

    def log_manual_attendance(self, enrollment_number: str, session: str = Config.DEFAULT_SESSION, confidence: float = 1.0) -> Dict[str, Any]:
        if not self.db_manager:
            return {"success": False, "error": "Database is not available"}
        student = self.db_manager.get_student_by_enrollment(enrollment_number)
        if not student:
            return {"success": False, "error": "Student not found"}
        logged = self.db_manager.log_attendance(student[0], session, confidence)
        return {"success": True, "logged": logged, "student": student}

    def list_registered_people(self) -> List[Dict[str, Any]]:
        return self.db_manager.list_registered_people() if self.db_manager else []

    def delete_registered_person(self, student_id: int) -> bool:
        if not self.db_manager:
            return False
        deleted = self.db_manager.delete_student(student_id)
        if deleted:
            self.load_known_students()
        return deleted

    def start_attendance_loop(self, session: str = Config.DEFAULT_SESSION, callback=None) -> bool:
        if cv2 is None:
            logger.error("OpenCV is not installed. Install requirements-minimal.txt.")
            return False
        if not self.is_initialized or not self.db_manager or not self.face_system:
            logger.error("System not initialized")
            return False
        camera = cv2.VideoCapture(Config.CAMERA_INDEX)
        if not camera.isOpened():
            logger.error("Cannot access camera %s", Config.CAMERA_INDEX)
            return False
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        camera.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)

        frame_count = 0
        last_processed_frame = None
        try:
            while True:
                ret, frame = camera.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % Config.RECOGNITION_INTERVAL == 0:
                    last_processed_frame = recognize_and_log(
                        frame,
                        self.known_students,
                        self.db_manager,
                        session=session,
                        tolerance=self.face_system.tolerance,
                        known_encodings=self.face_system.known_encodings,
                        known_names=self.face_system.known_names,
                        known_ids=self.face_system.known_ids,
                    )
                display_frame = last_processed_frame if last_processed_frame is not None else frame
                if callback:
                    callback(display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()
        return True

    def get_attendance_logs(self, date: Optional[str] = None) -> List[Any]:
        return self.db_manager.get_attendance_logs(date=date) if self.db_manager else []

    def export_logs_to_csv(self, filename: str, date: Optional[str] = None) -> bool:
        return self.db_manager.export_to_csv(filename, date=date) if self.db_manager else False


def print_check_report(check_camera: bool = True) -> bool:
    report = Config.dependency_report(check_camera=check_camera)
    print("Face Attendance Setup Check")
    print("=" * 32)
    print(f"Backend mode: {report['backend_mode']}")
    print(f"Python version: {report['python_version']}")
    print(f"CLI/lightweight ready: {'yes' if report['cli_ready'] else 'no'}")
    print(f"GUI ready: {'yes' if report['gui_ready'] else 'no'}")
    print(f"Full face recognition ready: {'yes' if report['full_ready'] else 'no'}")
    if not report["face_install_supported"]:
        print("Automatic face recognition install is skipped on this Python version to avoid dlib build failures.")
        print("Use Python 3.10 or 3.11 for full automatic recognition, or keep lightweight/manual mode here.")
    if report["missing_minimal"]:
        print("Missing minimal dependencies: " + ", ".join(report["missing_minimal"]))
    if report["missing_gui"]:
        print("Missing GUI dependencies: " + ", ".join(report["missing_gui"]))
        for name in report["missing_gui"]:
            detail = report["dependency_details"].get(name)
            if detail and detail != "ok":
                print(f"- {name}: {detail}")
        print("Use --mode cli, or install a Python build with Tk support for GUI mode.")
    if report["missing_full"]:
        print("Missing full-mode dependencies: " + ", ".join(report["missing_full"]))
    if report["camera_available"] is not None:
        print(f"Camera available: {'yes' if report['camera_available'] else 'no'}")
    print("\nConfiguration:")
    for key, value in Config.get_config_dict().items():
        print(f"- {key}: {value}")
    return bool(report["minimal_ready"])


def run_gui_application(database_path: Optional[str] = None) -> None:
    report = Config.dependency_report(check_camera=False)
    if not report["cli_ready"]:
        print(ErrorHandler.format_error(ErrorHandler.handle_dependency_error(report["missing_minimal"])))
        return
    if not report["gui_ready"]:
        print("GUI mode is not available in this Python environment.")
        for name in report["missing_gui"]:
            detail = report["dependency_details"].get(name)
            print(f"- {name}: {detail}")
        print("\nStarting CLI mode instead. Use Ctrl+C or the 'quit' command to exit.")
        run_command_line_interface(database_path)
        return
    try:
        import tkinter as tk
        from gui import AttendanceGUI
    except Exception as exc:
        print(f"Failed to load GUI: {exc}")
        print("\nStarting CLI mode instead. Use Ctrl+C or the 'quit' command to exit.")
        run_command_line_interface(database_path)
        return
    app = AttendanceSystem(database_path)
    root = tk.Tk()
    gui_app = AttendanceGUI(root)
    gui_app.set_system(app)
    root.mainloop()


def run_command_line_interface(database_path: Optional[str] = None) -> None:
    app = AttendanceSystem(database_path)
    status = app.get_system_status()
    print("Face Recognition Attendance System - CLI")
    print("=" * 48)
    print(f"System Status: {'Ready' if status.get('initialized') else 'Not Ready'}")
    print(f"Automatic face recognition: {'Ready' if status.get('full_ready') else 'Not installed'}")
    stats = status.get("database_stats", {})
    print(f"Total Students: {stats.get('total_students', 0)}")
    print(f"Total Attendance Records: {stats.get('total_attendance_records', 0)}")
    print("\nCommands: register, people, delete, manual, attendance, logs, export, check, quit")

    while True:
        command = input("\nEnter command: ").strip().lower()
        if command in {"quit", "exit"}:
            break
        if command == "check":
            print_check_report(check_camera=True)
        elif command == "register":
            name = input("Student name: ").strip()
            enrollment = input("Enrollment number (blank for auto): ").strip() or None
            category = input("Class/department (blank for none): ").strip() or None
            manual = input("Allow manual-only enrollment if face backend is unavailable? [y/N]: ").strip().lower() == "y"
            result = app.register_student(name, enrollment_number=enrollment, category=category, allow_manual_without_face=manual)
            print(result)
        elif command == "people":
            people = app.list_registered_people()
            if not people:
                print("No registered people yet.")
            for person in people:
                group = f" | {person['category']}" if person.get("category") else ""
                print(
                    f"{person['id']}: {person['name']} | {person['enrollment_number']}"
                    f"{group} | photos={person['image_count']} | face_samples={person['embedding_count']}"
                )
        elif command == "delete":
            person_id = input("Registered person ID to delete: ").strip()
            if not person_id.isdigit():
                print("Enter a numeric ID. Use the people command to find it.")
                continue
            confirm = input("Delete this registration and attendance records? [y/N]: ").strip().lower()
            if confirm == "y":
                print("Deleted" if app.delete_registered_person(int(person_id)) else "Delete failed")
        elif command == "manual":
            enrollment = input("Enrollment number: ").strip()
            session = input(f"Session ({Config.DEFAULT_SESSION}): ").strip() or Config.DEFAULT_SESSION
            print(app.log_manual_attendance(enrollment, session))
        elif command == "attendance":
            session = input(f"Session ({Config.DEFAULT_SESSION}): ").strip() or Config.DEFAULT_SESSION
            if not face_backend_available():
                print("Automatic recognition requires requirements-full.txt. Use the manual command for lightweight attendance.")
            else:
                app.start_attendance_loop(session)
        elif command == "logs":
            date = input("Date (YYYY-MM-DD, blank for all): ").strip() or None
            for log in app.get_attendance_logs(date)[:20]:
                print(f"{log[2]} | {log[1]} | {log[3]} | {log[4]:.2f}")
        elif command == "export":
            filename = input("CSV filename: ").strip()
            date = input("Date filter (YYYY-MM-DD, blank for all): ").strip() or None
            print("Exported" if app.export_logs_to_csv(filename, date) else "Export failed")
        else:
            print("Unknown command")


def main() -> None:
    parser = argparse.ArgumentParser(description="Face Recognition Attendance System")
    parser.add_argument("--mode", choices=["gui", "cli"], default="gui")
    parser.add_argument("--database", type=str, default=Config.DATABASE_PATH)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--check", action="store_true", help="Run setup diagnostics and exit")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.check:
        ok = print_check_report(check_camera=True)
        sys.exit(0 if ok else 1)

    try:
        if args.mode == "gui":
            run_gui_application(args.database)
        else:
            run_command_line_interface(args.database)
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
    except Exception as exc:
        logger.exception("Application error")
        print(f"Application failed: {exc}")


if __name__ == "__main__":
    main()
