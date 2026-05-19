"""
Logging utilities for the Face Recognition Attendance System.

This module provides centralized logging functionality with different
log levels and output formats for debugging and monitoring.
"""

import logging
import os
from datetime import datetime
from typing import Optional
import sys

from utils.config import Config


class AttendanceLogger:
    """Custom logger for the attendance system."""
    
    def __init__(self):
        """Initialize the logger with configuration settings."""
        self.logger = logging.getLogger('attendance_system')
        self.logger.setLevel(getattr(logging, Config.LOG_LEVEL))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up console and file handlers for logging."""
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if Config.LOG_FILE:
            # Ensure log directory exists
            log_dir = os.path.dirname(Config.LOG_FILE)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(Config.LOG_FILE)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, exc_info: bool = False):
        """Log debug message."""
        self.logger.debug(message, exc_info=exc_info)
    
    def info(self, message: str, exc_info: bool = False):
        """Log info message."""
        self.logger.info(message, exc_info=exc_info)
    
    def warning(self, message: str, exc_info: bool = False):
        """Log warning message."""
        self.logger.warning(message, exc_info=exc_info)
    
    def error(self, message: str, exc_info: bool = True):
        """Log error message."""
        self.logger.error(message, exc_info=exc_info)
    
    def critical(self, message: str, exc_info: bool = True):
        """Log critical message."""
        self.logger.critical(message, exc_info=exc_info)


# Create global logger instance
logger = AttendanceLogger()


def log_function_call(func):
    """
    Decorator to log function calls.
    
    Args:
        func: Function to be decorated
        
    Returns:
        Wrapped function with logging
    """
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling function: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Function {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Function {func.__name__} failed with error: {str(e)}")
            raise
    
    return wrapper


def log_performance(func):
    """
    Decorator to log function execution time.
    
    Args:
        func: Function to be decorated
        
    Returns:
        Wrapped function with performance logging
    """
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        logger.debug(f"Starting {func.__name__} at {start_time}")
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            logger.debug(f"{func.__name__} completed in {execution_time:.4f} seconds")
            return result
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            logger.error(f"{func.__name__} failed after {execution_time:.4f} seconds with error: {str(e)}")
            raise
    
    return wrapper


class LogContext:
    """Context manager for logging operations."""
    
    def __init__(self, operation_name: str):
        """
        Initialize log context.
        
        Args:
            operation_name (str): Name of the operation being logged
        """
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        """Enter the context and log start."""
        self.start_time = datetime.now()
        logger.info(f"Starting operation: {self.operation_name} at {self.start_time}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and log completion or error."""
        end_time = datetime.now()
        execution_time = (end_time - self.start_time).total_seconds()
        
        if exc_type is None:
            logger.info(f"Operation {self.operation_name} completed successfully in {execution_time:.4f} seconds")
        else:
            logger.error(f"Operation {self.operation_name} failed after {execution_time:.4f} seconds with error: {str(exc_val)}")
        
        return False  # Don't suppress exceptions


def log_database_operation(operation: str, table: str, success: bool = True, error: Optional[str] = None):
    """
    Log database operations.
    
    Args:
        operation (str): Type of database operation (INSERT, UPDATE, DELETE, SELECT)
        table (str): Table name
        success (bool): Whether the operation was successful
        error (str, optional): Error message if operation failed
    """
    if success:
        logger.info(f"Database {operation} operation on table '{table}' completed successfully")
    else:
        logger.error(f"Database {operation} operation on table '{table}' failed: {error}")


def log_face_recognition_event(event_type: str, student_name: Optional[str] = None, 
                              confidence: Optional[float] = None, success: bool = True):
    """
    Log face recognition events.
    
    Args:
        event_type (str): Type of face recognition event (DETECTION, RECOGNITION, TRAINING)
        student_name (str, optional): Student name if applicable
        confidence (float, optional): Recognition confidence score
        success (bool): Whether the event was successful
    """
    message = f"Face recognition {event_type} event"
    
    if student_name:
        message += f" for student '{student_name}'"
    
    if confidence is not None:
        message += f" with confidence {confidence:.4f}"
    
    if success:
        logger.info(message)
    else:
        logger.warning(f"{message} - FAILED")


def log_system_status(component: str, status: str, details: Optional[str] = None):
    """
    Log system component status.
    
    Args:
        component (str): Component name (CAMERA, DATABASE, GUI, RECOGNITION)
        status (str): Status (STARTED, STOPPED, ERROR, WARNING)
        details (str, optional): Additional details
    """
    message = f"System {component} status: {status}"
    if details:
        message += f" - {details}"
    
    if status in ['ERROR', 'CRITICAL']:
        logger.error(message)
    elif status == 'WARNING':
        logger.warning(message)
    else:
        logger.info(message)