"""
Helper utilities for the Face Recognition Attendance System.

This module provides various utility functions for image processing,
file operations, data validation, and other common tasks.
"""

import os
import cv2
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional, Union
import face_recognition
from PIL import Image
import logging

from utils.logger import logger
from utils.config import Config


def validate_image_path(image_path: str) -> bool:
    """
    Validate that an image path exists and is readable.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        bool: True if image is valid, False otherwise
    """
    try:
        if not os.path.exists(image_path):
            logger.warning(f"Image path does not exist: {image_path}")
            return False
        
        # Try to read the image
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Cannot read image file: {image_path}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error validating image path {image_path}: {str(e)}")
        return False


def save_image_with_timestamp(image: np.ndarray, prefix: str = "student") -> str:
    """
    Save an image with a timestamp in the filename.
    
    Args:
        image (np.ndarray): Image to save
        prefix (str): Prefix for the filename
        
    Returns:
        str: Path where the image was saved
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{timestamp}.jpg"
        filepath = os.path.join(Config.IMAGES_DIR, filename)
        
        # Ensure directory exists
        os.makedirs(Config.IMAGES_DIR, exist_ok=True)
        
        # Save image
        cv2.imwrite(filepath, image)
        logger.info(f"Image saved to: {filepath}")
        return filepath
    
    except Exception as e:
        logger.error(f"Error saving image: {str(e)}")
        raise


def resize_image(image: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    """
    Resize an image to specified dimensions.
    
    Args:
        image (np.ndarray): Input image
        width (int): Target width
        height (int): Target height
        
    Returns:
        np.ndarray: Resized image
    """
    try:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        return image


def convert_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to RGB format.
    
    Args:
        image (np.ndarray): BGR image
        
    Returns:
        np.ndarray: RGB image
    """
    try:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error(f"Error converting image to RGB: {str(e)}")
        return image


def get_face_locations(image: np.ndarray, method: str = "hog", upsamples: int = 1) -> List[Tuple[int, int, int, int]]:
    """
    Detect face locations in an image.
    
    Args:
        image (np.ndarray): Input image (RGB format)
        method (str): Detection method ("hog" or "cnn")
        upsamples (int): Number of times to upsample the image
        
    Returns:
        List[Tuple[int, int, int, int]]: List of face locations (top, right, bottom, left)
    """
    try:
        face_locations = face_recognition.face_locations(
            image, 
            number_of_times_to_upsample=upsamples, 
            model=method
        )
        logger.debug(f"Found {len(face_locations)} face(s) using {method} method")
        return face_locations
    except Exception as e:
        logger.error(f"Error detecting faces: {str(e)}")
        return []


def get_face_encodings(image: np.ndarray, face_locations: List[Tuple[int, int, int, int]] = None) -> List[np.ndarray]:
    """
    Get face encodings from an image.
    
    Args:
        image (np.ndarray): Input image (RGB format)
        face_locations (List[Tuple[int, int, int, int]], optional): Pre-detected face locations
        
    Returns:
        List[np.ndarray]: List of face encodings
    """
    try:
        face_encodings = face_recognition.face_encodings(image, face_locations)
        logger.debug(f"Generated {len(face_encodings)} face encoding(s)")
        return face_encodings
    except Exception as e:
        logger.error(f"Error generating face encodings: {str(e)}")
        return []


def compare_faces(known_encodings: List[np.ndarray], face_encoding: np.ndarray, 
                 tolerance: float = 0.6) -> List[bool]:
    """
    Compare a face encoding against known encodings.
    
    Args:
        known_encodings (List[np.ndarray]): List of known face encodings
        face_encoding (np.ndarray): Face encoding to compare
        tolerance (float): Distance threshold for matching
        
    Returns:
        List[bool]: List of match results
    """
    try:
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
        return matches
    except Exception as e:
        logger.error(f"Error comparing faces: {str(e)}")
        return []


def face_distance(known_encodings: List[np.ndarray], face_encoding: np.ndarray) -> List[float]:
    """
    Calculate face distances between encodings.
    
    Args:
        known_encodings (List[np.ndarray]): List of known face encodings
        face_encoding (np.ndarray): Face encoding to compare
        
    Returns:
        List[float]: List of face distances
    """
    try:
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        return distances
    except Exception as e:
        logger.error(f"Error calculating face distances: {str(e)}")
        return []


def draw_face_boxes(image: np.ndarray, face_locations: List[Tuple[int, int, int, int]], 
                   labels: List[str] = None, colors: List[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    Draw bounding boxes around detected faces.
    
    Args:
        image (np.ndarray): Input image
        face_locations (List[Tuple[int, int, int, int]]): List of face locations
        labels (List[str], optional): Labels for each face
        colors (List[Tuple[int, int, int]], optional): Colors for each bounding box
        
    Returns:
        np.ndarray: Image with face boxes drawn
    """
    try:
        result_image = image.copy()
        
        for i, (top, right, bottom, left) in enumerate(face_locations):
            # Set color
            color = colors[i] if colors and i < len(colors) else (0, 255, 0)
            
            # Draw rectangle
            cv2.rectangle(result_image, (left, top), (right, bottom), color, 2)
            
            # Add label if provided
            if labels and i < len(labels):
                label = labels[i]
                cv2.rectangle(result_image, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(result_image, label, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
        
        return result_image
    
    except Exception as e:
        logger.error(f"Error drawing face boxes: {str(e)}")
        return image


def validate_student_data(name: str, enrollment_number: str) -> Tuple[bool, str]:
    """
    Validate student registration data.
    
    Args:
        name (str): Student name
        enrollment_number (str): Enrollment number
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        # Check name
        if not name or not name.strip():
            return False, "Name cannot be empty"
        
        if len(name.strip()) < 2:
            return False, "Name must be at least 2 characters long"
        
        # Check enrollment number
        if not enrollment_number or not enrollment_number.strip():
            return False, "Enrollment number cannot be empty"
        
        if len(enrollment_number.strip()) < 3:
            return False, "Enrollment number must be at least 3 characters long"
        
        # Check for special characters in name (allow letters, spaces, hyphens, apostrophes)
        import re
        if not re.match(r"^[a-zA-Z\s\-']+$", name.strip()):
            return False, "Name contains invalid characters"
        
        return True, "Valid"
    
    except Exception as e:
        logger.error(f"Error validating student data: {str(e)}")
        return False, f"Validation error: {str(e)}"


def format_datetime(dt: datetime) -> str:
    """
    Format datetime object to readable string.
    
    Args:
        dt (datetime): Datetime object
        
    Returns:
        str: Formatted datetime string
    """
    try:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.error(f"Error formatting datetime: {str(e)}")
        return str(dt)


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        float: File size in MB
    """
    try:
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        return round(size_mb, 2)
    except Exception as e:
        logger.error(f"Error getting file size for {file_path}: {str(e)}")
        return 0.0


def cleanup_old_files(directory: str, max_age_days: int = 30):
    """
    Clean up old files in a directory.
    
    Args:
        directory (str): Directory to clean
        max_age_days (int): Maximum age of files in days
    """
    try:
        import time
        
        current_time = time.time()
        cutoff_time = current_time - (max_age_days * 24 * 60 * 60)
        
        if not os.path.exists(directory):
            return
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                file_mtime = os.path.getmtime(file_path)
                if file_mtime < cutoff_time:
                    os.remove(file_path)
                    logger.info(f"Removed old file: {file_path}")
    
    except Exception as e:
        logger.error(f"Error cleaning up old files in {directory}: {str(e)}")


def validate_camera_access(camera_index: int = 0) -> bool:
    """
    Validate that camera is accessible.
    
    Args:
        camera_index (int): Camera index
        
    Returns:
        bool: True if camera is accessible, False otherwise
    """
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error(f"Cannot access camera with index {camera_index}")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            logger.error(f"Cannot read frames from camera {camera_index}")
            return False
        
        logger.info(f"Camera {camera_index} is accessible")
        return True
    
    except Exception as e:
        logger.error(f"Error validating camera access: {str(e)}")
        return False


def get_system_info() -> dict:
    """
    Get system information for debugging.
    
    Returns:
        dict: System information
    """
    try:
        import platform
        import psutil
        
        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'opencv_version': cv2.__version__,
            'face_recognition_version': face_recognition.__version__,
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
            'camera_accessible': validate_camera_access()
        }
        
        return info
    
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        return {'error': str(e)}