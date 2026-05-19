"""
Face encoding module for the Face Recognition Attendance System.

This module handles face feature extraction using 128-dimensional embeddings
from the face_recognition library.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
import logging

from utils.logger import logger
from utils.helpers import get_face_encodings, face_distance, compare_faces
from utils.config import Config


class FaceEncoder:
    """Face encoding class for generating 128-dimensional face embeddings."""
    
    def __init__(self, num_jitters: int = 1, model: str = "small"):
        """
        Initialize the face encoder.
        
        Args:
            num_jitters (int): Number of times to jitter the image for better encoding
            model (str): Model to use ("small" for 5-point landmarks, "large" for 68-point)
        """
        self.num_jitters = num_jitters
        self.model = model
        
        logger.info(f"FaceEncoder initialized with jitters: {self.num_jitters}, model: {self.model}")
    
    def encode_face(self, image: np.ndarray, face_locations: Optional[List[Tuple[int, int, int, int]]] = None) -> List[np.ndarray]:
        """
        Generate face encodings from an image.
        
        Args:
            image (np.ndarray): Input image in RGB format
            face_locations (List[Tuple[int, int, int, int]], optional): Pre-detected face locations
            
        Returns:
            List[np.ndarray]: List of 128-dimensional face encodings
        """
        try:
            # Ensure image is in RGB format
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert BGR to RGB if needed
                if image.dtype == np.uint8:
                    if image[:,:,0].mean() > image[:,:,2].mean():
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            face_encodings = get_face_encodings(image, face_locations, self.num_jitters, self.model)
            
            logger.debug(f"Generated {len(face_encodings)} face encoding(s)")
            return face_encodings
        
        except Exception as e:
            logger.error(f"Error encoding faces: {str(e)}")
            return []
    
    def encode_single_face(self, image: np.ndarray, face_location: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Generate encoding for a single face.
        
        Args:
            image (np.ndarray): Input image in RGB format
            face_location (Tuple[int, int, int, int]): Single face location
            
        Returns:
            Optional[np.ndarray]: 128-dimensional face encoding or None if failed
        """
        try:
            encodings = self.encode_face(image, [face_location])
            return encodings[0] if encodings else None
        
        except Exception as e:
            logger.error(f"Error encoding single face: {str(e)}")
            return None
    
    def encode_faces_from_frame(self, frame: np.ndarray, face_locations: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """
        Encode faces from a video frame.
        
        Args:
            frame (np.ndarray): Input video frame (BGR format)
            face_locations (List[Tuple[int, int, int, int]]): List of face locations
            
        Returns:
            List[np.ndarray]: List of face encodings
        """
        try:
            # Convert BGR to RGB for encoding
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Encode faces
            face_encodings = self.encode_face(rgb_frame, face_locations)
            
            logger.debug(f"Encoded {len(face_encodings)} face(s) from frame")
            return face_encodings
        
        except Exception as e:
            logger.error(f"Error encoding faces from frame: {str(e)}")
            return []
    
    def set_num_jitters(self, num_jitters: int):
        """
        Set the number of jitters for encoding.
        
        Args:
            num_jitters (int): Number of times to jitter the image
        """
        if num_jitters >= 0:
            self.num_jitters = num_jitters
            logger.info(f"Number of jitters set to: {self.num_jitters}")
        else:
            logger.warning(f"Invalid num_jitters value: {num_jitters}. Using current value: {self.num_jitters}")
    
    def set_model(self, model: str):
        """
        Set the encoding model.
        
        Args:
            model (str): Model to use ("small" or "large")
        """
        if model.lower() in ["small", "large"]:
            self.model = model.lower()
            logger.info(f"Encoding model set to: {self.model}")
        else:
            logger.warning(f"Invalid model: {model}. Using current model: {self.model}")
    
    def get_encoding_info(self) -> dict:
        """
        Get current encoding configuration.
        
        Returns:
            dict: Encoding configuration information
        """
        return {
            'num_jitters': self.num_jitters,
            'model': self.model,
            'embedding_dimension': 128,
            'supported_models': ['small', 'large']
        }
    
    def validate_encoding_quality(self, encoding: np.ndarray) -> dict:
        """
        Validate the quality of a face encoding.
        
        Args:
            encoding (np.ndarray): Face encoding to validate
            
        Returns:
            dict: Quality metrics
        """
        try:
            if encoding is None or len(encoding) != 128:
                return {'valid': False, 'error': 'Invalid encoding dimension'}
            
            # Calculate encoding statistics
            encoding_stats = {
                'mean': float(np.mean(encoding)),
                'std': float(np.std(encoding)),
                'min': float(np.min(encoding)),
                'max': float(np.max(encoding)),
                'norm': float(np.linalg.norm(encoding))
            }
            
            # Check for quality indicators
            quality_metrics = {
                'valid': True,
                'dimension': len(encoding),
                'norm_range': 0.8 <= encoding_stats['norm'] <= 1.2,  # Normalized encodings should be around 1.0
                'value_range': -2.0 <= encoding_stats['min'] and encoding_stats['max'] <= 2.0,  # Reasonable value range
                'stats': encoding_stats
            }
            
            logger.debug(f"Encoding quality metrics: {quality_metrics}")
            return quality_metrics
        
        except Exception as e:
            logger.error(f"Error validating encoding quality: {str(e)}")
            return {'valid': False, 'error': str(e)}


class FaceMatcher:
    """Face matching class for comparing face encodings."""
    
    def __init__(self, tolerance: float = None):
        """
        Initialize the face matcher.
        
        Args:
            tolerance (float, optional): Distance threshold for matching
        """
        self.tolerance = tolerance or Config.RECOGNITION_THRESHOLD
        
        logger.info(f"FaceMatcher initialized with tolerance: {self.tolerance}")
    
    def match_faces(self, known_encodings: List[np.ndarray], face_encoding: np.ndarray) -> List[bool]:
        """
        Compare a face encoding against known encodings.
        
        Args:
            known_encodings (List[np.ndarray]): List of known face encodings
            face_encoding (np.ndarray): Face encoding to compare
            
        Returns:
            List[bool]: List of match results
        """
        try:
            matches = compare_faces(known_encodings, face_encoding, self.tolerance)
            return matches
        
        except Exception as e:
            logger.error(f"Error matching faces: {str(e)}")
            return []
    
    def calculate_distances(self, known_encodings: List[np.ndarray], face_encoding: np.ndarray) -> List[float]:
        """
        Calculate distances between encodings.
        
        Args:
            known_encodings (List[np.ndarray]): List of known face encodings
            face_encoding (np.ndarray): Face encoding to compare
            
        Returns:
            List[float]: List of face distances
        """
        try:
            distances = face_distance(known_encodings, face_encoding)
            return distances
        
        except Exception as e:
            logger.error(f"Error calculating face distances: {str(e)}")
            return []
    
    def find_best_match(self, known_encodings: List[np.ndarray], face_encoding: np.ndarray, 
                       known_names: Optional[List[str]] = None) -> Tuple[Optional[int], Optional[str], Optional[float]]:
        """
        Find the best match for a face encoding.
        
        Args:
            known_encodings (List[np.ndarray]): List of known face encodings
            face_encoding (np.ndarray): Face encoding to match
            known_names (List[str], optional): List of corresponding names
            
        Returns:
            Tuple[Optional[int], Optional[str], Optional[float]]: (best_match_index, best_match_name, best_distance)
        """
        try:
            if not known_encodings or face_encoding is None:
                return None, None, None
            
            # Calculate distances
            distances = self.calculate_distances(known_encodings, face_encoding)
            
            if not distances:
                return None, None, None
            
            # Find the closest match
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]
            
            # Check if it's a valid match based on tolerance
            if best_distance <= self.tolerance:
                best_match_name = known_names[best_match_index] if known_names else f"Person {best_match_index + 1}"
                return best_match_index, best_match_name, best_distance
            else:
                return None, "Unknown", best_distance
        
        except Exception as e:
            logger.error(f"Error finding best match: {str(e)}")
            return None, None, None
    
    def match_multiple_faces(self, known_encodings: List[np.ndarray], face_encodings: List[np.ndarray],
                           known_names: Optional[List[str]] = None) -> List[Tuple[Optional[int], Optional[str], Optional[float]]]:
        """
        Match multiple face encodings against known encodings.
        
        Args:
            known_encodings (List[np.ndarray]): List of known face encodings
            face_encodings (List[np.ndarray]): List of face encodings to match
            known_names (List[str], optional): List of corresponding names
            
        Returns:
            List[Tuple[Optional[int], Optional[str], Optional[float]]]: List of match results
        """
        try:
            matches = []
            
            for face_encoding in face_encodings:
                match_result = self.find_best_match(known_encodings, face_encoding, known_names)
                matches.append(match_result)
            
            return matches
        
        except Exception as e:
            logger.error(f"Error matching multiple faces: {str(e)}")
            return []
    
    def set_tolerance(self, tolerance: float):
        """
        Set the matching tolerance.
        
        Args:
            tolerance (float): Distance threshold for matching
        """
        if 0.0 <= tolerance <= 1.0:
            self.tolerance = tolerance
            logger.info(f"Matching tolerance set to: {self.tolerance}")
        else:
            logger.warning(f"Invalid tolerance value: {tolerance}. Using current tolerance: {self.tolerance}")
    
    def get_matching_info(self) -> dict:
        """
        Get current matching configuration.
        
        Returns:
            dict: Matching configuration information
        """
        return {
            'tolerance': self.tolerance,
            'tolerance_range': (0.0, 1.0),
            'description': 'Lower tolerance = stricter matching, Higher tolerance = more lenient matching'
        }


class FaceRecognitionEngine:
    """Complete face recognition engine combining detection and encoding."""
    
    def __init__(self, detection_method: str = "hog", upsamples: int = 1, 
                 num_jitters: int = 1, model: str = "small", tolerance: float = 0.6):
        """
        Initialize the face recognition engine.
        
        Args:
            detection_method (str): Face detection method
            upsamples (int): Number of upsamples for detection
            num_jitters (int): Number of jitters for encoding
            model (str): Encoding model
            tolerance (float): Matching tolerance
        """
        self.face_detector = FaceDetector(detection_method, upsamples)
        self.face_encoder = FaceEncoder(num_jitters, model)
        self.face_matcher = FaceMatcher(tolerance)
        
        logger.info("FaceRecognitionEngine initialized")
    
    def recognize_faces_in_frame(self, frame: np.ndarray, known_encodings: List[np.ndarray], 
                               known_names: List[str]) -> Tuple[List[Tuple[int, int, int, int]], List[str], List[float]]:
        """
        Complete face recognition pipeline for a video frame.
        
        Args:
            frame (np.ndarray): Input video frame (BGR format)
            known_encodings (List[np.ndarray]): List of known face encodings
            known_names (List[str]): List of corresponding names
            
        Returns:
            Tuple[List[Tuple[int, int, int, int]], List[str], List[float]]: 
                (face_locations, recognized_names, confidence_scores)
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = self.face_detector.detect_faces(rgb_frame)
            
            if not face_locations:
                return [], [], []
            
            # Encode faces
            face_encodings = self.face_encoder.encode_face(rgb_frame, face_locations)
            
            if not face_encodings:
                return face_locations, ["Unknown"] * len(face_locations), [0.0] * len(face_locations)
            
            # Match faces
            matches = self.face_matcher.match_multiple_faces(known_encodings, face_encodings, known_names)
            
            # Extract results
            recognized_names = []
            confidence_scores = []
            
            for match_result in matches:
                best_match_index, best_match_name, best_distance = match_result
                
                if best_match_name is not None:
                    recognized_names.append(best_match_name)
                    confidence = max(0.0, 1.0 - best_distance)  # Convert distance to confidence
                    confidence_scores.append(confidence)
                else:
                    recognized_names.append("Unknown")
                    confidence_scores.append(0.0)
            
            return face_locations, recognized_names, confidence_scores
        
        except Exception as e:
            logger.error(f"Error in face recognition pipeline: {str(e)}")
            return [], [], []
    
    def get_engine_info(self) -> dict:
        """
        Get complete engine configuration.
        
        Returns:
            dict: Engine configuration information
        """
        return {
            'detection': self.face_detector.get_detection_info(),
            'encoding': self.face_encoder.get_encoding_info(),
            'matching': self.face_matcher.get_matching_info()
        }
    
    def update_config(self, detection_method: Optional[str] = None, upsamples: Optional[int] = None,
                     num_jitters: Optional[int] = None, model: Optional[str] = None, 
                     tolerance: Optional[float] = None):
        """
        Update engine configuration.
        
        Args:
            detection_method (str, optional): New detection method
            upsamples (int, optional): New number of upsamples
            num_jitters (int, optional): New number of jitters
            model (str, optional): New encoding model
            tolerance (float, optional): New matching tolerance
        """
        if detection_method:
            self.face_detector.set_detection_method(detection_method)
        if upsamples is not None:
            self.face_detector.set_upsamples(upsamples)
        if num_jitters is not None:
            self.face_encoder.set_num_jitters(num_jitters)
        if model:
            self.face_encoder.set_model(model)
        if tolerance is not None:
            self.face_matcher.set_tolerance(tolerance)
        
        logger.info("FaceRecognitionEngine configuration updated")


def create_face_recognition_engine(config: dict = None) -> FaceRecognitionEngine:
    """
    Factory function to create a face recognition engine with custom configuration.
    
    Args:
        config (dict, optional): Configuration dictionary
        
    Returns:
        FaceRecognitionEngine: Configured face recognition engine
    """
    try:
        if config is None:
            config = {}
        
        engine = FaceRecognitionEngine(
            detection_method=config.get('detection_method', 'hog'),
            upsamples=config.get('upsamples', 1),
            num_jitters=config.get('num_jitters', 1),
            model=config.get('model', 'small'),
            tolerance=config.get('tolerance', 0.6)
        )
        
        logger.info("Created FaceRecognitionEngine with custom configuration")
        return engine
    
    except Exception as e:
        logger.error(f"Error creating face recognition engine: {str(e)}")
        return FaceRecognitionEngine()