"""
Face detection module for the Face Recognition Attendance System.

This module handles face detection using HOG or CNN-based methods
from the face_recognition library.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

from utils.logger import logger
from utils.helpers import get_face_locations, draw_face_boxes
from utils.config import Config


class FaceDetector:
    """Face detection class using face_recognition library."""
    
    def __init__(self, detection_method: str = None, upsamples: int = None):
        """
        Initialize the face detector.
        
        Args:
            detection_method (str, optional): Detection method ("hog" or "cnn")
            upsamples (int, optional): Number of times to upsample the image
        """
        self.detection_method = detection_method or Config.DETECTION_METHOD
        self.upsamples = upsamples or Config.UPSAMPLES
        
        logger.info(f"FaceDetector initialized with method: {self.detection_method}, upsamples: {self.upsamples}")
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.
        
        Args:
            image (np.ndarray): Input image in RGB format
            
        Returns:
            List[Tuple[int, int, int, int]]: List of face locations (top, right, bottom, left)
        """
        try:
            # Ensure image is in RGB format
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert BGR to RGB if needed
                if image.dtype == np.uint8:
                    # Assume BGR if values are high, RGB if values are low
                    if image[:,:,0].mean() > image[:,:,2].mean():
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            face_locations = get_face_locations(
                image, 
                method=self.detection_method, 
                upsamples=self.upsamples
            )
            
            logger.debug(f"Detected {len(face_locations)} face(s) in image")
            return face_locations
        
        except Exception as e:
            logger.error(f"Error detecting faces: {str(e)}")
            return []
    
    def detect_faces_in_frame(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
        """
        Detect faces in a video frame and return locations with processed frame.
        
        Args:
            frame (np.ndarray): Input video frame (BGR format)
            
        Returns:
            Tuple[List[Tuple[int, int, int, int]], np.ndarray]: Face locations and processed frame
        """
        try:
            # Convert BGR to RGB for face detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = self.detect_faces(rgb_frame)
            
            # Draw face boxes on the frame
            processed_frame = draw_face_boxes(frame.copy(), face_locations)
            
            return face_locations, processed_frame
        
        except Exception as e:
            logger.error(f"Error detecting faces in frame: {str(e)}")
            return [], frame
    
    def set_detection_method(self, method: str):
        """
        Set the detection method.
        
        Args:
            method (str): Detection method ("hog" or "cnn")
        """
        if method.lower() in ["hog", "cnn"]:
            self.detection_method = method.lower()
            logger.info(f"Detection method set to: {self.detection_method}")
        else:
            logger.warning(f"Invalid detection method: {method}. Using current method: {self.detection_method}")
    
    def set_upsamples(self, upsamples: int):
        """
        Set the number of upsamples.
        
        Args:
            upsamples (int): Number of times to upsample the image
        """
        if upsamples >= 0:
            self.upsamples = upsamples
            logger.info(f"Upsamples set to: {self.upsamples}")
        else:
            logger.warning(f"Invalid upsamples value: {upsamples}. Using current value: {self.upsamples}")
    
    def get_detection_info(self) -> dict:
        """
        Get current detection configuration.
        
        Returns:
            dict: Detection configuration information
        """
        return {
            'detection_method': self.detection_method,
            'upsamples': self.upsamples,
            'supported_methods': ['hog', 'cnn']
        }
    
    def validate_detection_performance(self, test_image: np.ndarray) -> dict:
        """
        Validate detection performance on a test image.
        
        Args:
            test_image (np.ndarray): Test image for performance validation
            
        Returns:
            dict: Performance metrics
        """
        try:
            import time
            
            # Measure detection time
            start_time = time.time()
            face_locations = self.detect_faces(test_image)
            detection_time = time.time() - start_time
            
            # Calculate performance metrics
            metrics = {
                'detection_time_seconds': detection_time,
                'faces_detected': len(face_locations),
                'image_shape': test_image.shape,
                'detection_method': self.detection_method,
                'upsamples': self.upsamples,
                'success': len(face_locations) > 0
            }
            
            logger.info(f"Detection performance: {metrics}")
            return metrics
        
        except Exception as e:
            logger.error(f"Error validating detection performance: {str(e)}")
            return {'error': str(e), 'success': False}


class MultiFaceDetector(FaceDetector):
    """Enhanced face detector with additional features for multi-face scenarios."""
    
    def __init__(self, detection_method: str = None, upsamples: int = None, min_face_size: int = 100):
        """
        Initialize the multi-face detector.
        
        Args:
            detection_method (str, optional): Detection method
            upsamples (int, optional): Number of upsamples
            min_face_size (int): Minimum face size in pixels
        """
        super().__init__(detection_method, upsamples)
        self.min_face_size = min_face_size
    
    def detect_faces_with_confidence(self, image: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect faces and estimate confidence based on face size.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            List[Tuple[Tuple[int, int, int, int], float]]: Face locations with confidence scores
        """
        try:
            face_locations = self.detect_faces(image)
            face_confidences = []
            
            for location in face_locations:
                top, right, bottom, left = location
                face_width = right - left
                face_height = bottom - top
                
                # Calculate confidence based on face size
                # Larger faces are more likely to be detected accurately
                min_dimension = min(face_width, face_height)
                confidence = min(1.0, min_dimension / self.min_face_size)
                
                face_confidences.append((location, confidence))
            
            # Sort by confidence (highest first)
            face_confidences.sort(key=lambda x: x[1], reverse=True)
            
            return face_confidences
        
        except Exception as e:
            logger.error(f"Error detecting faces with confidence: {str(e)}")
            return []
    
    def filter_faces_by_size(self, face_locations: List[Tuple[int, int, int, int]], 
                           min_size: int = None, max_size: int = None) -> List[Tuple[int, int, int, int]]:
        """
        Filter faces based on size constraints.
        
        Args:
            face_locations (List[Tuple[int, int, int, int]]): List of face locations
            min_size (int, optional): Minimum face size
            max_size (int, optional): Maximum face size
            
        Returns:
            List[Tuple[int, int, int, int]]: Filtered face locations
        """
        try:
            filtered_faces = []
            min_size = min_size or self.min_face_size
            max_size = max_size or float('inf')
            
            for location in face_locations:
                top, right, bottom, left = location
                face_width = right - left
                face_height = bottom - top
                min_dimension = min(face_width, face_height)
                
                if min_size <= min_dimension <= max_size:
                    filtered_faces.append(location)
            
            logger.debug(f"Filtered {len(face_locations)} faces to {len(filtered_faces)} based on size")
            return filtered_faces
        
        except Exception as e:
            logger.error(f"Error filtering faces by size: {str(e)}")
            return face_locations
    
    def detect_and_process_frame(self, frame: np.ndarray, show_confidence: bool = False) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
        """
        Detect faces in frame and return with enhanced visualization.
        
        Args:
            frame (np.ndarray): Input video frame
            show_confidence (bool): Whether to show confidence scores
            
        Returns:
            Tuple[List[Tuple[int, int, int, int]], np.ndarray]: Face locations and processed frame
        """
        try:
            # Convert to RGB for detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces with confidence
            face_confidences = self.detect_faces_with_confidence(rgb_frame)
            face_locations = [fc[0] for fc in face_confidences]
            
            # Prepare labels and colors
            labels = []
            colors = []
            
            for i, (location, confidence) in enumerate(face_confidences):
                if show_confidence:
                    label = f"Face {i+1} ({confidence:.2f})"
                else:
                    label = f"Face {i+1}"
                
                labels.append(label)
                
                # Color based on confidence
                if confidence > 0.8:
                    color = (0, 255, 0)  # Green for high confidence
                elif confidence > 0.5:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (0, 0, 255)  # Red for low confidence
                
                colors.append(color)
            
            # Draw face boxes
            processed_frame = draw_face_boxes(frame.copy(), face_locations, labels, colors)
            
            return face_locations, processed_frame
        
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return [], frame


def create_face_detector(method: str = "hog", upsamples: int = 1, multi_face: bool = False, 
                        min_face_size: int = 100) -> FaceDetector:
    """
    Factory function to create a face detector.
    
    Args:
        method (str): Detection method
        upsamples (int): Number of upsamples
        multi_face (bool): Whether to use multi-face detector
        min_face_size (int): Minimum face size for multi-face detector
        
    Returns:
        FaceDetector: Configured face detector instance
    """
    try:
        if multi_face:
            detector = MultiFaceDetector(method, upsamples, min_face_size)
        else:
            detector = FaceDetector(method, upsamples)
        
        logger.info(f"Created face detector: {type(detector).__name__} with method={method}, upsamples={upsamples}")
        return detector
    
    except Exception as e:
        logger.error(f"Error creating face detector: {str(e)}")
        return FaceDetector()