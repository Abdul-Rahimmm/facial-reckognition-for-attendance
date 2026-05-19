"""
Configuration module for the Face Recognition Attendance System.

This module provides centralized configuration management with
default values, validation, and environment variable support.
"""

import os
from typing import Dict, Any, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)


class Config:
    """Configuration class with validation and defaults."""
    
    # Database configuration
    DATABASE_PATH = os.getenv('ATTENDANCE_DB_PATH', 'data/attendance.db')
    DATABASE_TIMEOUT = float(os.getenv('ATTENDANCE_DB_TIMEOUT', '30.0'))
    
    # Face recognition configuration
    RECOGNITION_TOLERANCE = float(os.getenv('ATTENDANCE_TOLERANCE', '0.6'))
    RECOGNITION_MIN_CONFIDENCE = float(os.getenv('ATTENDANCE_MIN_CONFIDENCE', '0.5'))
    RECOGNITION_MODEL = os.getenv('ATTENDANCE_MODEL', 'hog')  # 'hog' or 'cnn'
    RECOGNITION_UPSAMPLES = int(os.getenv('ATTENDANCE_UPSAMPLES', '1'))
    
    # Camera configuration
    CAMERA_INDEX = int(os.getenv('ATTENDANCE_CAMERA_INDEX', '0'))
    CAMERA_WIDTH = int(os.getenv('ATTENDANCE_CAMERA_WIDTH', '640'))
    CAMERA_HEIGHT = int(os.getenv('ATTENDANCE_CAMERA_HEIGHT', '480'))
    CAMERA_FPS = int(os.getenv('ATTENDANCE_CAMERA_FPS', '30'))
    
    # GUI configuration
    GUI_WIDTH = int(os.getenv('ATTENDANCE_GUI_WIDTH', '1000'))
    GUI_HEIGHT = int(os.getenv('ATTENDANCE_GUI_HEIGHT', '700'))
    GUI_THEME_COLOR = os.getenv('ATTENDANCE_THEME_COLOR', '#f0f0f0')
    
    # Session configuration
    DEFAULT_SESSION = os.getenv('ATTENDANCE_DEFAULT_SESSION', 'default')
    SESSION_TIMEOUT = int(os.getenv('ATTENDANCE_SESSION_TIMEOUT', '300'))  # 5 minutes
    
    # File paths
    DATA_DIR = os.getenv('ATTENDANCE_DATA_DIR', 'data')
    IMAGES_DIR = os.getenv('ATTENDANCE_IMAGES_DIR', os.path.join(DATA_DIR, 'images'))
    LOGS_DIR = os.getenv('ATTENDANCE_LOGS_DIR', 'logs')
    LOG_FILE = os.getenv('ATTENDANCE_LOG_FILE', os.path.join(LOGS_DIR, 'attendance.log'))
    
    # Recognition thresholds
    MIN_FACE_SIZE = int(os.getenv('ATTENDANCE_MIN_FACE_SIZE', '100'))
    MAX_FACES_PER_FRAME = int(os.getenv('ATTENDANCE_MAX_FACES', '10'))
    
    # Validation ranges
    TOLERANCE_RANGE = (0.3, 0.8)
    CONFIDENCE_RANGE = (0.0, 1.0)
    CAMERA_INDEX_RANGE = (0, 10)
    IMAGE_COUNT_RANGE = (3, 20)
    
    @classmethod
    def validate_tolerance(cls, tolerance: float) -> bool:
        """Validate recognition tolerance is within acceptable range."""
        return cls.TOLERANCE_RANGE[0] <= tolerance <= cls.TOLERANCE_RANGE[1]
    
    @classmethod
    def validate_confidence(cls, confidence: float) -> bool:
        """Validate minimum confidence is within acceptable range."""
        return cls.CONFIDENCE_RANGE[0] <= confidence <= cls.CONFIDENCE_RANGE[1]
    
    @classmethod
    def validate_camera_index(cls, index: int) -> bool:
        """Validate camera index is within acceptable range."""
        return cls.CAMERA_INDEX_RANGE[0] <= index <= cls.CAMERA_INDEX_RANGE[1]
    
    @classmethod
    def validate_image_count(cls, count: int) -> bool:
        """Validate image count is within acceptable range."""
        return cls.IMAGE_COUNT_RANGE[0] <= count <= cls.IMAGE_COUNT_RANGE[1]
    
    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """Get all configuration values as a dictionary."""
        return {
            'database_path': cls.DATABASE_PATH,
            'database_timeout': cls.DATABASE_TIMEOUT,
            'recognition_tolerance': cls.RECOGNITION_TOLERANCE,
            'recognition_min_confidence': cls.RECOGNITION_MIN_CONFIDENCE,
            'recognition_model': cls.RECOGNITION_MODEL,
            'recognition_upsamples': cls.RECOGNITION_UPSAMPLES,
            'camera_index': cls.CAMERA_INDEX,
            'camera_width': cls.CAMERA_WIDTH,
            'camera_height': cls.CAMERA_HEIGHT,
            'camera_fps': cls.CAMERA_FPS,
            'gui_width': cls.GUI_WIDTH,
            'gui_height': cls.GUI_HEIGHT,
            'gui_theme_color': cls.GUI_THEME_COLOR,
            'default_session': cls.DEFAULT_SESSION,
            'session_timeout': cls.SESSION_TIMEOUT,
            'data_dir': cls.DATA_DIR,
            'images_dir': cls.IMAGES_DIR,
            'logs_dir': cls.LOGS_DIR,
            'log_file': cls.LOG_FILE,
            'min_face_size': cls.MIN_FACE_SIZE,
            'max_faces_per_frame': cls.MAX_FACES_PER_FRAME,
        }
    
    @classmethod
    def update_tolerance(cls, new_tolerance: float) -> bool:
        """Update the recognition tolerance with validation."""
        if not cls.validate_tolerance(new_tolerance):
            logger.warning(f"Invalid tolerance value: {new_tolerance}. Must be between {cls.TOLERANCE_RANGE[0]} and {cls.TOLERANCE_RANGE[1]}")
            return False
        
        cls.RECOGNITION_TOLERANCE = new_tolerance
        logger.info(f"Recognition tolerance updated to: {new_tolerance}")
        return True
    
    @classmethod
    def update_min_confidence(cls, new_confidence: float) -> bool:
        """Update the minimum confidence with validation."""
        if not cls.validate_confidence(new_confidence):
            logger.warning(f"Invalid confidence value: {new_confidence}. Must be between {cls.CONFIDENCE_RANGE[0]} and {cls.CONFIDENCE_RANGE[1]}")
            return False
        
        cls.RECOGNITION_MIN_CONFIDENCE = new_confidence
        logger.info(f"Minimum confidence updated to: {new_confidence}")
        return True
    
    @classmethod
    def update_camera_index(cls, new_index: int) -> bool:
        """Update the camera index with validation."""
        if not cls.validate_camera_index(new_index):
            logger.warning(f"Invalid camera index: {new_index}. Must be between {cls.CAMERA_INDEX_RANGE[0]} and {cls.CAMERA_INDEX_RANGE[1]}")
            return False
        
        cls.CAMERA_INDEX = new_index
        logger.info(f"Camera index updated to: {new_index}")
        return True
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        directories = [
            cls.DATA_DIR,
            cls.IMAGES_DIR,
            cls.LOGS_DIR
        ]
        
        for directory in directories:
            if directory:  # Skip empty directory names
                os.makedirs(directory, exist_ok=True)
                logger.debug(f"Ensured directory exists: {directory}")


class ErrorHandler:
    """Centralized error handling with detailed messages and recovery suggestions."""
    
    @staticmethod
    def handle_camera_error(error: Exception, context: str = "") -> Dict[str, Any]:
        """Handle camera-related errors."""
        error_msg = str(error)
        logger.error(f"Camera error in {context}: {error_msg}")
        
        suggestions = []
        
        if "Cannot access camera" in error_msg or "No cameras found" in error_msg:
            suggestions = [
                "Check if camera is properly connected",
                "Ensure no other applications are using the camera",
                "Try different camera index (0, 1, 2, etc.)",
                "Check camera permissions in your operating system"
            ]
            error_type = "NO_CAMERA"
        
        elif "Failed to capture frame" in error_msg:
            suggestions = [
                "Camera connection may have been lost",
                "Try restarting the camera or application",
                "Check camera drivers and firmware"
            ]
            error_type = "CAPTURE_FAILED"
        
        else:
            suggestions = [
                "Check camera connection and permissions",
                "Try restarting the application",
                "Contact support if problem persists"
            ]
            error_type = "CAMERA_UNKNOWN"
        
        return {
            'error_type': error_type,
            'error_message': error_msg,
            'suggestions': suggestions,
            'context': context,
            'recovery_possible': True
        }
    
    @staticmethod
    def handle_face_detection_error(error: Exception, context: str = "") -> Dict[str, Any]:
        """Handle face detection errors."""
        error_msg = str(error)
        logger.error(f"Face detection error in {context}: {error_msg}")
        
        suggestions = [
            "Ensure proper lighting conditions",
            "Make sure faces are clearly visible",
            "Check that people are within camera range",
            "Try adjusting camera angle or distance"
        ]
        
        return {
            'error_type': 'FACE_DETECTION_FAILED',
            'error_message': error_msg,
            'suggestions': suggestions,
            'context': context,
            'recovery_possible': True
        }
    
    @staticmethod
    def handle_database_error(error: Exception, context: str = "") -> Dict[str, Any]:
        """Handle database-related errors."""
        error_msg = str(error)
        logger.error(f"Database error in {context}: {error_msg}")
        
        suggestions = []
        
        if "UNIQUE constraint failed" in error_msg:
            suggestions = [
                "Student with this enrollment number already exists",
                "Use a different enrollment number",
                "Check for duplicate entries"
            ]
            error_type = "DUPLICATE_ENTRY"
        
        elif "database is locked" in error_msg:
            suggestions = [
                "Database may be in use by another process",
                "Wait a moment and try again",
                "Check for proper file permissions"
            ]
            error_type = "DATABASE_LOCKED"
        
        elif "No such table" in error_msg:
            suggestions = [
                "Database tables may not be initialized",
                "Run database initialization",
                "Check database file integrity"
            ]
            error_type = "TABLE_NOT_FOUND"
        
        else:
            suggestions = [
                "Check database file permissions",
                "Ensure database file is not corrupted",
                "Try restarting the application"
            ]
            error_type = "DATABASE_UNKNOWN"
        
        return {
            'error_type': error_type,
            'error_message': error_msg,
            'suggestions': suggestions,
            'context': context,
            'recovery_possible': True
        }
    
    @staticmethod
    def handle_recognition_error(error: Exception, context: str = "") -> Dict[str, Any]:
        """Handle face recognition errors."""
        error_msg = str(error)
        logger.error(f"Recognition error in {context}: {error_msg}")
        
        suggestions = [
            "Ensure faces are clearly visible",
            "Check lighting conditions",
            "Make sure person is registered in the system",
            "Try adjusting recognition tolerance settings"
        ]
        
        return {
            'error_type': 'RECOGNITION_FAILED',
            'error_message': error_msg,
            'suggestions': suggestions,
            'context': context,
            'recovery_possible': True
        }
    
    @staticmethod
    def handle_low_confidence(confidence: float, threshold: float, context: str = "") -> Dict[str, Any]:
        """Handle low confidence recognition results."""
        logger.warning(f"Low confidence recognition in {context}: {confidence:.3f} (threshold: {threshold:.3f})")
        
        suggestions = [
            "Ensure good lighting conditions",
            "Make sure face is clearly visible",
            "Check camera focus and angle",
            "Consider lowering recognition threshold",
            "Retrain with more images if needed"
        ]
        
        return {
            'error_type': 'LOW_CONFIDENCE',
            'confidence': confidence,
            'threshold': threshold,
            'suggestions': suggestions,
            'context': context,
            'recovery_possible': True
        }
    
    @staticmethod
    def handle_duplicate_name(name: str, context: str = "") -> Dict[str, Any]:
        """Handle duplicate name scenarios."""
        logger.warning(f"Duplicate name detected in {context}: {name}")
        
        suggestions = [
            "Use full names to avoid conflicts",
            "Add middle names or initials",
            "Use enrollment numbers for disambiguation",
            "Consider using unique identifiers"
        ]
        
        return {
            'error_type': 'DUPLICATE_NAME',
            'name': name,
            'suggestions': suggestions,
            'context': context,
            'recovery_possible': True
        }
    
    @staticmethod
    def show_user_error(error_info: Dict[str, Any], parent=None):
        """Display user-friendly error message."""
        import tkinter as tk
        from tkinter import messagebox
        
        error_type = error_info.get('error_type', 'UNKNOWN')
        error_message = error_info.get('error_message', 'An unknown error occurred')
        suggestions = error_info.get('suggestions', [])
        
        # Create user-friendly message
        message = f"Error Type: {error_type}\n\n"
        message += f"Details: {error_message}\n\n"
        message += "Suggestions:\n"
        for i, suggestion in enumerate(suggestions, 1):
            message += f"{i}. {suggestion}\n"
        
        message += "\nDo you want to continue or exit the application?"
        
        # Show error dialog
        if parent:
            result = messagebox.showerror("System Error", message, parent=parent)
        else:
            result = messagebox.showerror("System Error", message)
        
        return result


# Initialize directories on import
Config.ensure_directories()