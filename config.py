"""
Configuration and user-facing error helpers for the attendance system.
"""

import importlib.util
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        logger.warning("Invalid float for %s; using %s", name, default)
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        logger.warning("Invalid integer for %s; using %s", name, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _module_import_status(module_name: str) -> Tuple[bool, str]:
    if importlib.util.find_spec(module_name) is None:
        return False, "not installed"
    try:
        __import__(module_name)
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


class Config:
    """Central app configuration sourced from environment variables."""

    DATA_DIR = os.getenv("ATTENDANCE_DATA_DIR", "data")
    IMAGES_DIR = os.getenv("ATTENDANCE_IMAGES_DIR", os.path.join(DATA_DIR, "images"))
    LOGS_DIR = os.getenv("ATTENDANCE_LOGS_DIR", "logs")
    DATABASE_PATH = os.getenv("ATTENDANCE_DB_PATH", os.path.join(DATA_DIR, "attendance.db"))
    DATABASE_TIMEOUT = _env_float("ATTENDANCE_DB_TIMEOUT", 30.0)
    LOG_FILE = os.getenv("ATTENDANCE_LOG_FILE", os.path.join(LOGS_DIR, "attendance.log"))

    BACKEND_MODE = os.getenv("ATTENDANCE_BACKEND", "auto").strip().lower()
    LOW_POWER_MODE = _env_bool("ATTENDANCE_LOW_POWER", True)
    RECOGNITION_TOLERANCE = _env_float("ATTENDANCE_TOLERANCE", 0.6)
    RECOGNITION_MIN_CONFIDENCE = _env_float("ATTENDANCE_MIN_CONFIDENCE", 0.5)
    RECOGNITION_MODEL = os.getenv("ATTENDANCE_MODEL", "hog").strip().lower()
    RECOGNITION_UPSAMPLES = _env_int("ATTENDANCE_UPSAMPLES", 0 if LOW_POWER_MODE else 1)
    RECOGNITION_INTERVAL = max(1, _env_int("ATTENDANCE_RECOGNITION_INTERVAL", 5 if LOW_POWER_MODE else 2))
    FRAME_SCALE = min(1.0, max(0.2, _env_float("ATTENDANCE_FRAME_SCALE", 0.5 if LOW_POWER_MODE else 1.0)))
    DUPLICATE_LOG_WINDOW_SECONDS = max(0, _env_int("ATTENDANCE_DUPLICATE_LOG_WINDOW", 0))

    CAMERA_INDEX = _env_int("ATTENDANCE_CAMERA_INDEX", 0)
    CAMERA_WIDTH = _env_int("ATTENDANCE_CAMERA_WIDTH", 640)
    CAMERA_HEIGHT = _env_int("ATTENDANCE_CAMERA_HEIGHT", 480)
    CAMERA_FPS = _env_int("ATTENDANCE_CAMERA_FPS", 15 if LOW_POWER_MODE else 30)

    GUI_WIDTH = _env_int("ATTENDANCE_GUI_WIDTH", 1000)
    GUI_HEIGHT = _env_int("ATTENDANCE_GUI_HEIGHT", 700)
    GUI_THEME_COLOR = os.getenv("ATTENDANCE_THEME_COLOR", "#f0f0f0")

    DEFAULT_SESSION = os.getenv("ATTENDANCE_DEFAULT_SESSION", "default")
    MIN_FACE_SIZE = _env_int("ATTENDANCE_MIN_FACE_SIZE", 60 if LOW_POWER_MODE else 100)
    MAX_FACES_PER_FRAME = _env_int("ATTENDANCE_MAX_FACES", 10)

    TOLERANCE_RANGE = (0.3, 0.8)
    CONFIDENCE_RANGE = (0.0, 1.0)
    CAMERA_INDEX_RANGE = (0, 10)
    IMAGE_COUNT_RANGE = (1, 20)
    VALID_BACKENDS = {"auto", "minimal", "full"}
    FACE_RECOGNITION_MAX_PYTHON = (3, 11)

    @classmethod
    def ensure_directories(cls) -> None:
        for directory in (cls.DATA_DIR, cls.IMAGES_DIR, cls.LOGS_DIR):
            if directory:
                os.makedirs(directory, exist_ok=True)

    @classmethod
    def validate_tolerance(cls, tolerance: float) -> bool:
        return cls.TOLERANCE_RANGE[0] <= tolerance <= cls.TOLERANCE_RANGE[1]

    @classmethod
    def validate_confidence(cls, confidence: float) -> bool:
        return cls.CONFIDENCE_RANGE[0] <= confidence <= cls.CONFIDENCE_RANGE[1]

    @classmethod
    def validate_camera_index(cls, index: int) -> bool:
        return cls.CAMERA_INDEX_RANGE[0] <= index <= cls.CAMERA_INDEX_RANGE[1]

    @classmethod
    def validate_image_count(cls, count: int) -> bool:
        return cls.IMAGE_COUNT_RANGE[0] <= count <= cls.IMAGE_COUNT_RANGE[1]

    @classmethod
    def validate_backend_mode(cls, backend: str) -> bool:
        return backend in cls.VALID_BACKENDS

    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        return {
            "database_path": cls.DATABASE_PATH,
            "backend_mode": cls.BACKEND_MODE,
            "low_power_mode": cls.LOW_POWER_MODE,
            "recognition_tolerance": cls.RECOGNITION_TOLERANCE,
            "recognition_min_confidence": cls.RECOGNITION_MIN_CONFIDENCE,
            "recognition_interval": cls.RECOGNITION_INTERVAL,
            "frame_scale": cls.FRAME_SCALE,
            "duplicate_log_window_seconds": cls.DUPLICATE_LOG_WINDOW_SECONDS,
            "camera_index": cls.CAMERA_INDEX,
            "camera_width": cls.CAMERA_WIDTH,
            "camera_height": cls.CAMERA_HEIGHT,
            "camera_fps": cls.CAMERA_FPS,
            "default_session": cls.DEFAULT_SESSION,
        }

    @classmethod
    def dependency_report(cls, check_camera: bool = False) -> Dict[str, Any]:
        python_version = sys.version_info[:3]
        face_install_supported = python_version[:2] <= cls.FACE_RECOGNITION_MAX_PYTHON
        tkinter_ok, tkinter_detail = _module_import_status("tkinter")
        cv2_ok, cv2_detail = _module_import_status("cv2")
        numpy_ok, numpy_detail = _module_import_status("numpy")
        pil_ok, pil_detail = _module_import_status("PIL")
        face_ok, face_detail = _module_import_status("face_recognition")
        gui_ready = tkinter_ok and cv2_ok and numpy_ok and pil_ok
        cli_ready = cv2_ok and numpy_ok and pil_ok
        required_gui = {
            "tkinter": tkinter_ok,
            "cv2": cv2_ok,
            "numpy": numpy_ok,
            "PIL": pil_ok,
        }
        dependency_details = {
            "tkinter": tkinter_detail,
            "cv2": cv2_detail,
            "numpy": numpy_detail,
            "PIL": pil_detail,
            "face_recognition": face_detail,
        }
        full_ready = gui_ready and face_ok

        camera_available = None
        if check_camera and cv2_ok:
            try:
                import cv2  # type: ignore

                camera = cv2.VideoCapture(cls.CAMERA_INDEX)
                camera_available = camera.isOpened()
                camera.release()
            except Exception:
                camera_available = False

        missing_minimal = [name for name, ok in {"cv2": cv2_ok, "numpy": numpy_ok, "PIL": pil_ok}.items() if not ok]
        missing_gui = [name for name, ok in required_gui.items() if not ok]
        missing_full = list(missing_minimal)
        if not tkinter_ok:
            missing_full.append("tkinter")
        if not face_ok:
            missing_full.append("face_recognition")

        return {
            "backend_mode": cls.BACKEND_MODE,
            "python_version": ".".join(str(part) for part in python_version),
            "face_install_supported": face_install_supported,
            "minimal_ready": cli_ready,
            "cli_ready": cli_ready,
            "gui_ready": gui_ready,
            "full_ready": full_ready,
            "face_backend_available": face_ok,
            "required": required_gui,
            "dependency_details": dependency_details,
            "missing_minimal": missing_minimal,
            "missing_gui": missing_gui,
            "missing_full": missing_full,
            "camera_available": camera_available,
        }


class ErrorHandler:
    """Build concise, recoverable error messages for UI and CLI flows."""

    @staticmethod
    def handle_camera_error(error: Exception, context: str = "") -> Dict[str, Any]:
        return {
            "error_type": "CAMERA_UNAVAILABLE",
            "error_message": str(error),
            "suggestions": [
                "Check camera permissions.",
                "Close other apps using the camera.",
                "Try ATTENDANCE_CAMERA_INDEX=1 if another camera is connected.",
            ],
            "context": context,
            "recovery_possible": True,
        }

    @staticmethod
    def handle_dependency_error(missing: List[str], full_mode: bool = False) -> Dict[str, Any]:
        install_file = "requirements-full.txt" if full_mode else "requirements-minimal.txt"
        suggestions = [
            f"Install with: python -m pip install -r {install_file}",
            "Use ATTENDANCE_BACKEND=minimal on weak devices.",
            "Run python main.py --check after installing dependencies.",
        ]
        if full_mode and not Config.dependency_report()["face_install_supported"]:
            suggestions.insert(1, "Automatic face recognition needs Python 3.10 or 3.11; Python 3.12+ uses lightweight mode safely.")
        return {
            "error_type": "MISSING_DEPENDENCIES",
            "error_message": "Missing dependencies: " + ", ".join(missing),
            "suggestions": suggestions,
            "context": "startup",
            "recovery_possible": True,
        }

    @staticmethod
    def format_error(error_info: Dict[str, Any]) -> str:
        suggestions = "\n".join(f"- {item}" for item in error_info.get("suggestions", []))
        return f"{error_info.get('error_type', 'ERROR')}: {error_info.get('error_message', '')}\n{suggestions}"


Config.ensure_directories()
