"""
Recognition module for the Face Recognition Attendance System.

This module provides face detection, recognition, and image capture
functionality using the face_recognition library and OpenCV.
"""

import cv2
import face_recognition
import numpy as np
import pickle
import logging
import time
from typing import List, Tuple, Optional, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)


class FaceRecognitionSystem:
    """High-level face recognition system for attendance management."""
    
    def __init__(self, tolerance: float = 0.6, camera_index: int = 0):
        """
        Initialize the face recognition system.
        
        Args:
            tolerance (float): Recognition tolerance (lower = more strict)
            camera_index (int): Camera index for video capture
        """
        self.tolerance = tolerance
        self.camera_index = camera_index
        self.known_students = []
        self.known_encodings = []
        self.known_names = []
        self.known_ids = []
    
    def update_known_students(self, students: List[Tuple]):
        """
        Update the known students database.
        
        Args:
            students (List[Tuple]): List of students with embeddings
        """
        self.known_students = students
        self.known_encodings = []
        self.known_names = []
        self.known_ids = []
        
        for student_id, name, enrollment_number, embeddings in students:
            if embeddings:
                # Use average of all embeddings for this student
                avg_embedding = self.get_average_embedding(embeddings)
                if avg_embedding:
                    self.known_encodings.append(avg_embedding)
                    self.known_names.append(name)
                    self.known_ids.append(student_id)
        
        logger.info(f"Updated known students: {len(self.known_students)} students, {len(self.known_encodings)} encodings")
    
    def get_average_embedding(self, embeddings: List[List[float]]) -> Optional[List[float]]:
        """
        Calculate average embedding from multiple embeddings.
        
        Args:
            embeddings (List[List[float]]): List of face embeddings
            
        Returns:
            Optional[List[float]]: Average embedding or None if empty
        """
        if not embeddings:
            return None
        
        try:
            # Convert to numpy array for calculation
            embeddings_array = np.array(embeddings)
            average_embedding = np.mean(embeddings_array, axis=0)
            return average_embedding.tolist()
        except Exception as e:
            logger.error(f"Error calculating average embedding: {e}")
            return None
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """
        Validate face embedding format.
        
        Args:
            embedding (List[float]): Face embedding to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not isinstance(embedding, list):
            return False
        
        if len(embedding) != 128:  # face_recognition uses 128-dimensional embeddings
            return False
        
        # Check if all values are numeric
        try:
            float_values = [float(x) for x in embedding]
            return True
        except (ValueError, TypeError):
            return False
    
    def serialize_embedding(self, embedding: List[float]) -> bytes:
        """
        Serialize face embedding for database storage.
        
        Args:
            embedding (List[float]): Face embedding to serialize
            
        Returns:
            bytes: Serialized embedding
        """
        return pickle.dumps(embedding)
    
    def deserialize_embedding(self, serialized_embedding: bytes) -> List[float]:
        """
        Deserialize face embedding from database storage.
        
        Args:
            serialized_embedding (bytes): Serialized embedding
            
        Returns:
            List[float]: Deserialized embedding
        """
        return pickle.loads(serialized_embedding)


def detect_faces(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in an image using HOG-based detection.
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        List[Tuple[int, int, int, int]]: List of face locations (top, right, bottom, left)
    """
    try:
        # Convert BGR to RGB (OpenCV uses BGR, face_recognition uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        
        logger.debug(f"Detected {len(face_locations)} faces")
        return face_locations
    
    except Exception as e:
        logger.error(f"Error detecting faces: {e}")
        return []


def extract_embeddings(image: np.ndarray, face_locations: List[Tuple[int, int, int, int]]) -> List[List[float]]:
    """
    Extract face embeddings from detected faces.
    
    Args:
        image (np.ndarray): Input image
        face_locations (List[Tuple[int, int, int, int]]): List of face locations
        
    Returns:
        List[List[float]]: List of 128-dimensional face embeddings
    """
    try:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract embeddings
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        logger.debug(f"Extracted {len(face_encodings)} face embeddings")
        return face_encodings
    
    except Exception as e:
        logger.error(f"Error extracting face embeddings: {e}")
        return []


def match_face(unknown_encoding: List[float], known_encodings: List[List[float]], 
               known_names: List[str], tolerance: float = 0.6) -> Tuple[Optional[str], Optional[float]]:
    """
    Match an unknown face against known faces.
    
    Args:
        unknown_encoding (List[float]): Unknown face encoding
        known_encodings (List[List[float]]): List of known face encodings
        known_names (List[str]): List of known names
        tolerance (float): Matching tolerance
        
    Returns:
        Tuple[Optional[str], Optional[float]]: (matched_name, confidence) or (None, None)
    """
    try:
        if not known_encodings or not unknown_encoding:
            return None, None
        
        # Compare faces
        matches = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=tolerance)
        face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)
        
        if not matches:
            return None, None
        
        # Find the best match
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            name = known_names[best_match_index]
            confidence = 1.0 - face_distances[best_match_index]
            return name, confidence
        
        return None, None
    
    except Exception as e:
        logger.error(f"Error matching face: {e}")
        return None, None


def capture_images(name: str, num_images: int = 5) -> List[List[float]]:
    """
    Capture multiple images of a person for training.
    
    Args:
        name (str): Person's name
        num_images (int): Number of images to capture
        
    Returns:
        List[List[float]]: List of face embeddings
    """
    try:
        logger.info(f"Starting image capture for {name}")
        
        # Open camera
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            logger.error("Cannot access camera")
            return []
        
        embeddings = []
        captured_count = 0
        
        # Create window for display
        cv2.namedWindow("Face Capture - Press SPACE to capture, ESC to exit")
        
        while captured_count < num_images:
            ret, frame = camera.read()
            if not ret:
                logger.error("Failed to capture frame")
                break
            
            # Display frame
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Capturing images for: {name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Captured: {captured_count}/{num_images}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Detect faces in current frame
            face_locations = detect_faces(frame)
            
            # Draw rectangles around faces
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            cv2.imshow("Face Capture - Press SPACE to capture, ESC to exit", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space bar to capture
                if len(face_locations) == 1:
                    # Extract embedding
                    embeddings_list = extract_embeddings(frame, face_locations)
                    
                    if embeddings_list:
                        embeddings.append(embeddings_list[0])
                        captured_count += 1
                        logger.info(f"Captured image {captured_count}/{num_images} for {name}")
                        
                        # Show confirmation
                        temp_frame = frame.copy()
                        cv2.putText(temp_frame, "CAPTURED!", (50, 50),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        cv2.imshow("Face Capture - Press SPACE to capture, ESC to exit", temp_frame)
                        cv2.waitKey(500)  # Show message for 0.5 seconds
                    else:
                        logger.warning("No face detected in captured image")
                else:
                    logger.warning(f"Please ensure only one face is visible. Detected {len(face_locations)} faces.")
            
            elif key == 27:  # ESC to exit
                break
        
        # Cleanup
        camera.release()
        cv2.destroyAllWindows()
        
        logger.info(f"Completed image capture for {name}. Captured {len(embeddings)} images.")
        return embeddings
    
    except Exception as e:
        logger.error(f"Error during image capture: {e}")
        return []


def recognize_and_log(frame: np.ndarray, known_students: List[Tuple], 
                     db_manager: Any, session: str = "default", 
                     tolerance: float = 0.6) -> np.ndarray:
    """
    Recognize faces in a frame and log attendance.
    
    Args:
        frame (np.ndarray): Input video frame
        known_students (List[Tuple]): List of known students with embeddings
        db_manager (Any): Database manager instance
        session (str): Session identifier
        tolerance (float): Recognition tolerance
        
    Returns:
        np.ndarray: Processed frame with annotations
    """
    try:
        # Create a copy for processing
        processed_frame = frame.copy()
        
        # Detect faces
        face_locations = detect_faces(frame)
        
        if not face_locations:
            cv2.putText(processed_frame, "No faces detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return processed_frame
        
        # Extract embeddings
        face_encodings = extract_embeddings(frame, face_locations)
        
        if not face_encodings:
            cv2.putText(processed_frame, "No face encodings found", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return processed_frame
        
        # Prepare known data
        known_encodings = []
        known_names = []
        known_ids = []
        
        for student_id, name, enrollment_number, embeddings in known_students:
            if embeddings:
                avg_embedding = np.mean(np.array(embeddings), axis=0)
                known_encodings.append(avg_embedding)
                known_names.append(name)
                known_ids.append(student_id)
        
        # Process each face
        for (face_location, face_encoding) in zip(face_locations, face_encodings):
            top, right, bottom, left = face_location
            
            # Match face
            matched_name, confidence = match_face(
                face_encoding, known_encodings, known_names, tolerance
            )
            
            # Draw rectangle and label
            if matched_name and confidence is not None:
                # Successful recognition
                color = (0, 255, 0)  # Green
                label = f"{matched_name} ({confidence:.2f})"
                
                # Log attendance
                student_index = known_names.index(matched_name)
                student_id = known_ids[student_index]
                
                # Check if already logged recently (to avoid duplicates)
                # This is a simple check - in production, you might want more sophisticated duplicate prevention
                db_manager.log_attendance(student_id, session, confidence)
                
            else:
                # Unknown face
                color = (0, 0, 255)  # Red
                label = "Unknown"
            
            # Draw rectangle
            cv2.rectangle(processed_frame, (left, top), (right, bottom), color, 2)
            
            # Draw label
            cv2.rectangle(processed_frame, (left, bottom - 20), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(processed_frame, label, (left + 6, bottom - 6),
                       font, 0.5, (255, 255, 255), 1)
        
        return processed_frame
    
    except Exception as e:
        logger.error(f"Error in recognize_and_log: {e}")
        return frame


def get_system_status() -> Dict[str, Any]:
    """
    Get face recognition system status.
    
    Returns:
        Dict[str, Any]: System status information
    """
    try:
        # Check if camera is available
        camera_available = False
        try:
            camera = cv2.VideoCapture(0)
            if camera.isOpened():
                ret, frame = camera.read()
                if ret:
                    camera_available = True
                camera.release()
        except Exception:
            pass
        
        # Check if face_recognition is working
        face_recognition_working = False
        try:
            # Create a simple test image
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            face_locations = detect_faces(test_image)
            face_recognition_working = True
        except Exception:
            pass
        
        return {
            'camera_available': camera_available,
            'face_recognition_working': face_recognition_working,
            'opencv_version': cv2.__version__,
            'face_recognition_available': True
        }
    
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {
            'camera_available': False,
            'face_recognition_working': False,
            'error': str(e)
        }