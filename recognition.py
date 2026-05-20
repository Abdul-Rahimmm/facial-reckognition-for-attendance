"""
Face recognition helpers with a graceful minimal mode.

The module imports without OpenCV or face_recognition installed, which lets
`python main.py --check` explain what is missing instead of crashing at import.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config import Config

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    import face_recognition  # type: ignore
except Exception:
    face_recognition = None  # type: ignore

logger = logging.getLogger(__name__)


def computer_vision_available() -> bool:
    return cv2 is not None and np is not None


def face_backend_available() -> bool:
    if Config.BACKEND_MODE == "minimal":
        return False
    return computer_vision_available() and face_recognition is not None


def _put_text(frame: Any, text: str, y: int = 30) -> Any:
    if cv2 is not None and frame is not None:
        try:
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        except Exception:
            logger.debug("Could not draw status text on this frame type")
    return frame


class FaceRecognitionSystem:
    """Caches known encodings and exposes recognition readiness."""

    def __init__(self, tolerance: float = 0.6, camera_index: int = 0):
        self.tolerance = tolerance
        self.camera_index = camera_index
        self.known_students: List[Tuple] = []
        self.known_encodings: List[Any] = []
        self.known_names: List[str] = []
        self.known_ids: List[int] = []
        self.backend_available = face_backend_available()

    def update_known_students(self, students: List[Tuple]) -> None:
        self.known_students = students
        self.known_encodings = []
        self.known_names = []
        self.known_ids = []
        for student_id, name, _enrollment_number, embeddings in students:
            avg_embedding = self.get_average_embedding(embeddings)
            if avg_embedding is not None:
                self.known_encodings.append(avg_embedding)
                self.known_names.append(name)
                self.known_ids.append(student_id)
        logger.debug("Cached %s known face encoding(s)", len(self.known_encodings))

    def get_average_embedding(self, embeddings: List[Sequence[float]]) -> Optional[Any]:
        if not embeddings or np is None:
            return None
        try:
            return np.mean(np.asarray(embeddings, dtype="float64"), axis=0)
        except Exception as exc:
            logger.warning("Could not average embeddings: %s", exc)
            return None

    def validate_embedding(self, embedding: Sequence[Any]) -> bool:
        try:
            values = [float(value) for value in embedding]
            return len(values) == 128
        except Exception:
            return False

    def recognize_and_log(self, frame: Any, db_manager: Any, session: str = "default") -> Any:
        return recognize_and_log(
            frame,
            self.known_students,
            db_manager,
            session=session,
            tolerance=self.tolerance,
            known_encodings=self.known_encodings,
            known_names=self.known_names,
            known_ids=self.known_ids,
        )


def _scaled_frame(image: Any, scale: Optional[float] = None) -> Tuple[Any, float]:
    scale = Config.FRAME_SCALE if scale is None else scale
    if cv2 is None or scale >= 0.99:
        return image, 1.0
    return cv2.resize(image, (0, 0), fx=scale, fy=scale), scale


def _rescale_locations(locations: List[Tuple[int, int, int, int]], scale: float) -> List[Tuple[int, int, int, int]]:
    if scale == 1.0:
        return locations
    factor = 1.0 / scale
    return [
        (int(top * factor), int(right * factor), int(bottom * factor), int(left * factor))
        for top, right, bottom, left in locations
    ]


def detect_faces(image: Any, scale: Optional[float] = None) -> List[Tuple[int, int, int, int]]:
    if not face_backend_available():
        logger.debug("Face backend unavailable; skipping detection")
        return []
    try:
        small_image, used_scale = _scaled_frame(image, scale)
        rgb_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(
            rgb_image,
            number_of_times_to_upsample=Config.RECOGNITION_UPSAMPLES,
            model=Config.RECOGNITION_MODEL,
        )
        return _rescale_locations(locations[: Config.MAX_FACES_PER_FRAME], used_scale)
    except Exception as exc:
        logger.error("Error detecting faces: %s", exc)
        return []


def extract_embeddings(image: Any, face_locations: List[Tuple[int, int, int, int]]) -> List[Any]:
    if not face_backend_available() or not face_locations:
        return []
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return list(face_recognition.face_encodings(rgb_image, face_locations))
    except Exception as exc:
        logger.error("Error extracting face embeddings: %s", exc)
        return []


def match_face(
    unknown_encoding: Any,
    known_encodings: Sequence[Any],
    known_names: Sequence[str],
    tolerance: float = 0.6,
) -> Tuple[Optional[int], Optional[str], Optional[float]]:
    if not face_backend_available() or not known_encodings:
        return None, None, None
    try:
        distances = face_recognition.face_distance(known_encodings, unknown_encoding)
        if len(distances) == 0:
            return None, None, None
        best_index = int(np.argmin(distances))
        best_distance = float(distances[best_index])
        if best_distance <= tolerance:
            confidence = max(0.0, min(1.0, 1.0 - best_distance))
            return best_index, known_names[best_index], confidence
        return None, None, None
    except Exception as exc:
        logger.error("Error matching face: %s", exc)
        return None, None, None


def capture_images(name: str, num_images: int = 5, camera_index: Optional[int] = None) -> List[Any]:
    if not face_backend_available():
        logger.warning("Automatic face registration requires requirements-full.txt")
        return []
    camera = cv2.VideoCapture(Config.CAMERA_INDEX if camera_index is None else camera_index)
    if not camera.isOpened():
        logger.error("Cannot access camera")
        return []
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)

    embeddings: List[Any] = []
    window = "Face Capture - SPACE to capture, ESC to exit"
    cv2.namedWindow(window)
    try:
        while len(embeddings) < num_images:
            ret, frame = camera.read()
            if not ret:
                break
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Capturing: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"{len(embeddings)}/{num_images}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            face_locations = detect_faces(frame)
            for top, right, bottom, left in face_locations:
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.imshow(window, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and len(face_locations) == 1:
                encodings = extract_embeddings(frame, face_locations)
                if encodings:
                    embeddings.append(encodings[0])
            elif key == 27:
                break
    finally:
        camera.release()
        cv2.destroyWindow(window)
    return embeddings


def _known_data_from_students(known_students: List[Tuple]) -> Tuple[List[Any], List[str], List[int]]:
    system = FaceRecognitionSystem(Config.RECOGNITION_TOLERANCE, Config.CAMERA_INDEX)
    system.update_known_students(known_students)
    return system.known_encodings, system.known_names, system.known_ids


def recognize_and_log(
    frame: Any,
    known_students: List[Tuple],
    db_manager: Any,
    session: str = "default",
    tolerance: float = 0.6,
    known_encodings: Optional[List[Any]] = None,
    known_names: Optional[List[str]] = None,
    known_ids: Optional[List[int]] = None,
) -> Any:
    processed_frame = frame.copy() if hasattr(frame, "copy") else frame
    if cv2 is None:
        return processed_frame
    if not face_backend_available():
        return _put_text(processed_frame, "Automatic recognition unavailable")

    if known_encodings is None or known_names is None or known_ids is None:
        known_encodings, known_names, known_ids = _known_data_from_students(known_students)
    if not known_encodings:
        return _put_text(processed_frame, "No trained faces available")

    face_locations = detect_faces(frame)
    face_encodings = extract_embeddings(frame, face_locations)
    if not face_locations:
        return _put_text(processed_frame, "No faces detected")

    for face_location, face_encoding in zip(face_locations, face_encodings):
        top, right, bottom, left = face_location
        best_index, matched_name, confidence = match_face(face_encoding, known_encodings, known_names, tolerance)
        if best_index is not None and matched_name and confidence is not None and confidence >= Config.RECOGNITION_MIN_CONFIDENCE:
            color = (0, 255, 0)
            label = f"{matched_name} ({confidence:.2f})"
            db_manager.log_attendance(known_ids[best_index], session, confidence)
        else:
            color = (0, 0, 255)
            label = "Unknown"

        cv2.rectangle(processed_frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(processed_frame, (left, bottom - 22), (right, bottom), color, cv2.FILLED)
        cv2.putText(processed_frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
    return processed_frame


def get_system_status(check_camera: bool = False) -> Dict[str, Any]:
    report = Config.dependency_report(check_camera=check_camera)
    return {
        "camera_available": report["camera_available"],
        "opencv_available": cv2 is not None,
        "opencv_version": getattr(cv2, "__version__", None) if cv2 is not None else None,
        "numpy_available": np is not None,
        "face_recognition_available": face_recognition is not None,
        "face_recognition_working": face_backend_available(),
    }
