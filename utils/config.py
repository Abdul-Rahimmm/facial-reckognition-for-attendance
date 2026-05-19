"""
Configuration settings for the Face Recognition Attendance System.

This module contains all configurable parameters for the system including
recognition thresholds, database paths, image paths, and other settings.
"""

import os
from typing import Dict, Any


class Config:
    """Configuration class for the attendance system."""
    
    # Database configuration
    DATABASE_PATH = "data/attendance.db"
    
    # Image storage paths
    IMAGES_DIR = "data/images"
    EMBEDDINGS_DIR = "data/embeddings"
    
    # Face recognition settings
    RECOGNITION_THRESHOLD = 0.6  # Default threshold for face matching
    TOLERANCE_RANGE = (0.3, 0.8)  # Valid range for recognition threshold
    
    # Camera settings
    CAMERA_INDEX = 0  # Default camera index
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS = 30
    
    # Face detection settings
    DETECTION_METHOD = "hog"  # Options: "hog", "cnn"
    UPSAMPLES = 1  # Number of times to upsample the image
    
    # GUI settings
    WINDOW_TITLE = "Face Recognition Attendance System"
    WINDOW_WIDTH = 1000
    WINDOW_HEIGHT = 700
    THEME_COLOR = "#2c3e50"
    ACCENT_COLOR = "#3498db"
    
    # Logging settings
    LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
    LOG_FILE = "logs/attendance.log"
    
    # Export settings
    EXPORT_FORMATS = ["csv", "excel"]
    DEFAULT_EXPORT_FORMAT = "csv"
    
    # System behavior
    MAX_RECENT_ATTENDANCE = 50  # Maximum recent attendance records to display
    AUTO_SAVE_INTERVAL = 300  # Auto-save interval in seconds (5 minutes)
    
    @classmethod
    def validate_threshold(cls, threshold: float) -> bool:
        """
        Validate that the recognition threshold is within acceptable range.
        
        Args:
            threshold (float): Recognition threshold value
            
        Returns:
            bool: True if valid, False otherwise
        """
        return cls.TOLERANCE_RANGE[0] <= threshold <= cls.TOLERANCE_RANGE[1]
    
    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """
        Get all configuration values as a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary of configuration values
        """
        return {
            'database_path': cls.DATABASE_PATH,
            'images_dir': cls.IMAGES_DIR,
            'embeddings_dir': cls.EMBEDDINGS_DIR,
            'recognition_threshold': cls.RECOGNITION_THRESHOLD,
            'tolerance_range': cls.TOLERANCE_RANGE,
            'camera_index': cls.CAMERA_INDEX,
            'frame_width': cls.FRAME_WIDTH,
            'frame_height': cls.FRAME_HEIGHT,
            'fps': cls.FPS,
            'detection_method': cls.DETECTION_METHOD,
            'upsamples': cls.UPSAMPLES,
            'window_title': cls.WINDOW_TITLE,
            'window_width': cls.WINDOW_WIDTH,
            'window_height': cls.WINDOW_HEIGHT,
            'theme_color': cls.THEME_COLOR,
            'accent_color': cls.ACCENT_COLOR,
            'log_level': cls.LOG_LEVEL,
            'log_file': cls.LOG_FILE,
            'export_formats': cls.EXPORT_FORMATS,
            'default_export_format': cls.DEFAULT_EXPORT_FORMAT,
            'max_recent_attendance': cls.MAX_RECENT_ATTENDANCE,
            'auto_save_interval': cls.AUTO_SAVE_INTERVAL,
        }
    
    @classmethod
    def update_threshold(cls, new_threshold: float):
        """
        Update the recognition threshold.
        
        Args:
            new_threshold (float): New threshold value
            
        Raises:
            ValueError: If threshold is outside valid range
        """
        if not cls.validate_threshold(new_threshold):
            raise ValueError(f"Threshold must be between {cls.TOLERANCE_RANGE[0]} and {cls.TOLERANCE_RANGE[1]}")
        cls.RECOGNITION_THRESHOLD = new_threshold
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        directories = [
            os.path.dirname(cls.DATABASE_PATH),
            cls.IMAGES_DIR,
            cls.EMBEDDINGS_DIR,
            os.path.dirname(cls.LOG_FILE)
        ]
        
        for directory in directories:
            if directory:  # Skip empty directory names
                os.makedirs(directory, exist_ok=True)


# Initialize directories on import
Config.ensure_directories()