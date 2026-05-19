"""
Training module for the Face Recognition Attendance System.

This module handles the training process for registering new students
and computing their face embeddings for recognition.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
import os
from datetime import datetime

from utils.logger import logger
from utils.helpers import get_face_encodings, get_face_locations, validate_image_path
from utils.config import Config
from database.models import DatabaseManager
from recognition.face_encoder import FaceEncoder


class FaceTrainer:
    """Face training class for registering new students and computing embeddings."""
    
    def __init__(self, database_manager: DatabaseManager, num_jitters: int = 10, model: str = "small"):
        """
        Initialize the face trainer.
        
        Args:
            database_manager (DatabaseManager): Database manager instance
            num_jitters (int): Number of jitters for encoding (higher = more accurate but slower)
            model (str): Encoding model to use
        """
        self.db_manager = database_manager
        self.face_encoder = FaceEncoder(num_jitters, model)
        self.num_jitters = num_jitters
        self.model = model
        
        logger.info(f"FaceTrainer initialized with {num_jitters} jitters, model: {model}")
    
    def train_from_image(self, image_path: str, student_id: int) -> Dict:
        """
        Train face recognition from a single image.
        
        Args:
            image_path (str): Path to the training image
            student_id (int): ID of the student
            
        Returns:
            Dict: Training result with success status and details
        """
        try:
            # Validate image path
            if not validate_image_path(image_path):
                return {
                    'success': False,
                    'error': 'Invalid image path or unreadable image',
                    'student_id': student_id,
                    'encodings_created': 0
                }
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'success': False,
                    'error': 'Failed to load image',
                    'student_id': student_id,
                    'encodings_created': 0
                }
            
            # Convert to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = get_face_locations(rgb_image)
            
            if not face_locations:
                return {
                    'success': False,
                    'error': 'No faces detected in image',
                    'student_id': student_id,
                    'encodings_created': 0
                }
            
            if len(face_locations) > 1:
                logger.warning(f"Multiple faces detected in training image. Using first face only.")
            
            # Encode face
            face_encodings = self.face_encoder.encode_face(rgb_image, [face_locations[0]])
            
            if not face_encodings:
                return {
                    'success': False,
                    'error': 'Failed to generate face encoding',
                    'student_id': student_id,
                    'encodings_created': 0
                }
            
            # Register encoding
            success = self.db_manager.add_face_embedding(student_id, face_encodings[0])
            
            result = {
                'success': success,
                'student_id': student_id,
                'encodings_created': 1 if success else 0,
                'face_locations_found': len(face_locations),
                'image_path': image_path
            }
            
            if success:
                logger.info(f"Successfully trained face for student {student_id} from image {image_path}")
            else:
                result['error'] = 'Failed to save encoding to database'
            
            return result
        
        except Exception as e:
            logger.error(f"Error training from image: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'student_id': student_id,
                'encodings_created': 0
            }
    
    def train_from_multiple_images(self, image_paths: List[str], student_id: int) -> Dict:
        """
        Train face recognition from multiple images of the same person.
        
        Args:
            image_paths (List[str]): List of paths to training images
            student_id (int): ID of the student
            
        Returns:
            Dict: Training result with success status and details
        """
        try:
            successful_encodings = 0
            total_faces_detected = 0
            errors = []
            
            for image_path in image_paths:
                try:
                    # Train from single image
                    result = self.train_from_image(image_path, student_id)
                    
                    if result['success']:
                        successful_encodings += result['encodings_created']
                        total_faces_detected += result['face_locations_found']
                    else:
                        errors.append(f"{image_path}: {result.get('error', 'Unknown error')}")
                
                except Exception as e:
                    errors.append(f"{image_path}: {str(e)}")
                    logger.error(f"Error processing image {image_path}: {str(e)}")
            
            result = {
                'success': successful_encodings > 0,
                'student_id': student_id,
                'encodings_created': successful_encodings,
                'total_faces_detected': total_faces_detected,
                'images_processed': len(image_paths),
                'errors': errors
            }
            
            logger.info(f"Training completed for student {student_id}: {successful_encodings} encodings from {len(image_paths)} images")
            return result
        
        except Exception as e:
            logger.error(f"Error training from multiple images: {str(e)}")
            return {
                'success': False,
                'student_id': student_id,
                'encodings_created': 0,
                'error': str(e)
            }
    
    def train_from_video_frame(self, frame: np.ndarray, student_id: int) -> Dict:
        """
        Train face recognition from a video frame.
        
        Args:
            frame (np.ndarray): Video frame (BGR format)
            student_id (int): ID of the student
            
        Returns:
            Dict: Training result with success status and details
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = get_face_locations(rgb_frame)
            
            if not face_locations:
                return {
                    'success': False,
                    'error': 'No faces detected in frame',
                    'student_id': student_id,
                    'encodings_created': 0
                }
            
            if len(face_locations) > 1:
                logger.warning(f"Multiple faces detected in training frame. Using first face only.")
            
            # Encode face
            face_encodings = self.face_encoder.encode_face(rgb_frame, [face_locations[0]])
            
            if not face_encodings:
                return {
                    'success': False,
                    'error': 'Failed to generate face encoding',
                    'student_id': student_id,
                    'encodings_created': 0
                }
            
            # Register encoding
            success = self.db_manager.add_face_embedding(student_id, face_encodings[0])
            
            result = {
                'success': success,
                'student_id': student_id,
                'encodings_created': 1 if success else 0,
                'face_locations_found': len(face_locations)
            }
            
            if success:
                logger.info(f"Successfully trained face for student {student_id} from video frame")
            else:
                result['error'] = 'Failed to save encoding to database'
            
            return result
        
        except Exception as e:
            logger.error(f"Error training from video frame: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'student_id': student_id,
                'encodings_created': 0
            }
    
    def train_student_interactive(self, student_id: int, camera_index: int = 0, 
                                num_samples: int = 5) -> Dict:
        """
        Interactive training from live camera feed.
        
        Args:
            student_id (int): ID of the student
            camera_index (int): Camera index to use
            num_samples (int): Number of face samples to capture
            
        Returns:
            Dict: Training result with success status and details
        """
        try:
            # Validate camera access
            if not self._validate_camera(camera_index):
                return {
                    'success': False,
                    'error': f'Cannot access camera {camera_index}',
                    'student_id': student_id,
                    'samples_captured': 0
                }
            
            # Open camera
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                return {
                    'success': False,
                    'error': 'Failed to open camera',
                    'student_id': student_id,
                    'samples_captured': 0
                }
            
            captured_samples = 0
            training_results = []
            
            logger.info(f"Starting interactive training for student {student_id}. Capture {num_samples} samples.")
            
            while captured_samples < num_samples:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Display instructions
                cv2.putText(frame, f"Training: Capture {num_samples - captured_samples} more samples", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Face Training - Press SPACE to capture, ESC to exit', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC key
                    break
                elif key == 32:  # SPACE key
                    # Capture sample
                    result = self.train_from_video_frame(frame, student_id)
                    if result['success']:
                        captured_samples += 1
                        training_results.append(result)
                        logger.info(f"Sample {captured_samples} captured successfully")
                    else:
                        logger.warning(f"Failed to capture sample: {result.get('error', 'Unknown error')}")
            
            cap.release()
            cv2.destroyAllWindows()
            
            result = {
                'success': captured_samples > 0,
                'student_id': student_id,
                'samples_captured': captured_samples,
                'total_requested': num_samples,
                'training_results': training_results
            }
            
            logger.info(f"Interactive training completed for student {student_id}: {captured_samples}/{num_samples} samples captured")
            return result
        
        except Exception as e:
            logger.error(f"Error in interactive training: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'student_id': student_id,
                'samples_captured': 0
            }
    
    def validate_training_quality(self, student_id: int) -> Dict:
        """
        Validate the quality of training data for a student.
        
        Args:
            student_id (int): ID of the student
            
        Returns:
            Dict: Validation results with quality metrics
        """
        try:
            # Get student's face embeddings
            embeddings = self.db_manager.get_face_embeddings(student_id)
            
            if not embeddings:
                return {
                    'valid': False,
                    'student_id': student_id,
                    'error': 'No face embeddings found for student',
                    'quality_score': 0.0,
                    'recommendations': ['Register at least one face image for this student']
                }
            
            # Validate each embedding
            valid_embeddings = []
            invalid_count = 0
            
            for i, embedding in enumerate(embeddings):
                validation = self.face_encoder.validate_encoding_quality(embedding)
                if validation['valid']:
                    valid_embeddings.append(embedding)
                else:
                    invalid_count += 1
                    logger.warning(f"Invalid embedding {i} for student {student_id}: {validation.get('error', 'Unknown error')}")
            
            # Calculate quality metrics
            quality_score = 0.0
            recommendations = []
            
            if valid_embeddings:
                # Calculate average embedding statistics
                avg_embedding = np.mean(valid_embeddings, axis=0)
                embedding_stats = self.face_encoder.validate_encoding_quality(avg_embedding)
                
                # Quality score based on number of valid embeddings
                quality_score = min(1.0, len(valid_embeddings) / 3.0)  # Max score at 3+ embeddings
                
                # Generate recommendations
                if len(valid_embeddings) < 2:
                    recommendations.append("Add more face images for better recognition accuracy")
                if len(valid_embeddings) < 1:
                    recommendations.append("At least one valid face image is required")
                if invalid_count > 0:
                    recommendations.append(f"Remove {invalid_count} invalid face images")
            
            result = {
                'valid': len(valid_embeddings) > 0,
                'student_id': student_id,
                'total_embeddings': len(embeddings),
                'valid_embeddings': len(valid_embeddings),
                'invalid_embeddings': invalid_count,
                'quality_score': quality_score,
                'recommendations': recommendations
            }
            
            logger.info(f"Training validation for student {student_id}: {result}")
            return result
        
        except Exception as e:
            logger.error(f"Error validating training quality: {str(e)}")
            return {
                'valid': False,
                'student_id': student_id,
                'error': str(e),
                'quality_score': 0.0
            }
    
    def get_training_statistics(self) -> Dict:
        """
        Get training statistics for the entire system.
        
        Returns:
            Dict: Training statistics
        """
        try:
            # Get all students
            students = self.db_manager.get_all_students()
            
            total_students = len(students)
            students_with_faces = 0
            total_encodings = 0
            avg_encodings_per_student = 0.0
            
            for student in students:
                student_id = student[0]
                embeddings = self.db_manager.get_face_embeddings(student_id)
                if embeddings:
                    students_with_faces += 1
                    total_encodings += len(embeddings)
            
            if students_with_faces > 0:
                avg_encodings_per_student = total_encodings / students_with_faces
            
            stats = {
                'total_students': total_students,
                'students_with_faces': students_with_faces,
                'total_face_encodings': total_encodings,
                'average_encodings_per_student': avg_encodings_per_student,
                'coverage_percentage': (students_with_faces / total_students * 100) if total_students > 0 else 0.0,
                'training_completeness': 'Good' if students_with_faces == total_students else 'Incomplete'
            }
            
            logger.info(f"Training statistics: {stats}")
            return stats
        
        except Exception as e:
            logger.error(f"Error getting training statistics: {str(e)}")
            return {'error': str(e)}
    
    def _validate_camera(self, camera_index: int) -> bool:
        """Validate camera access."""
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                return False
            
            ret, frame = cap.read()
            cap.release()
            
            return ret and frame is not None
        except:
            return False


class TrainingManager:
    """High-level manager for training operations."""
    
    def __init__(self, database_manager: DatabaseManager):
        """
        Initialize the training manager.
        
        Args:
            database_manager (DatabaseManager): Database manager instance
        """
        self.db_manager = database_manager
        self.face_trainer = FaceTrainer(database_manager)
        
        logger.info("TrainingManager initialized")
    
    def register_new_student(self, name: str, enrollment_number: str, 
                           image_paths: List[str] = None, camera_samples: int = 0) -> Dict:
        """
        Register a new student with face training.
        
        Args:
            name (str): Student name
            enrollment_number (str): Student enrollment number
            image_paths (List[str], optional): List of training image paths
            camera_samples (int): Number of camera samples to capture
            
        Returns:
            Dict: Registration result
        """
        try:
            # Validate student data
            is_valid, error_msg = self._validate_student_data(name, enrollment_number)
            if not is_valid:
                return {
                    'success': False,
                    'error': error_msg,
                    'student_id': None
                }
            
            # Check if student already exists
            existing_student = self.db_manager.get_student_by_enrollment(enrollment_number)
            if existing_student:
                return {
                    'success': False,
                    'error': f"Student with enrollment number {enrollment_number} already exists",
                    'student_id': existing_student[0]
                }
            
            # Add student to database
            student_id = self.db_manager.add_student(name, enrollment_number)
            
            if not student_id:
                return {
                    'success': False,
                    'error': 'Failed to add student to database',
                    'student_id': None
                }
            
            training_result = {
                'success': True,
                'student_id': student_id,
                'name': name,
                'enrollment_number': enrollment_number,
                'training_details': {}
            }
            
            # Train from images if provided
            if image_paths:
                image_result = self.face_trainer.train_from_multiple_images(image_paths, student_id)
                training_result['training_details']['image_training'] = image_result
            
            # Train from camera if requested
            if camera_samples > 0:
                camera_result = self.face_trainer.train_student_interactive(
                    student_id, num_samples=camera_samples
                )
                training_result['training_details']['camera_training'] = camera_result
            
            logger.info(f"Successfully registered student {name} (ID: {student_id})")
            return training_result
        
        except Exception as e:
            logger.error(f"Error registering new student: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'student_id': None
            }
    
    def update_student_training(self, student_id: int, image_paths: List[str] = None, 
                               camera_samples: int = 0) -> Dict:
        """
        Update training data for an existing student.
        
        Args:
            student_id (int): ID of the student
            image_paths (List[str], optional): List of new training image paths
            camera_samples (int): Number of new camera samples to capture
            
        Returns:
            Dict: Update result
        """
        try:
            # Validate student exists
            student = self.db_manager.get_student_by_id(student_id)
            if not student:
                return {
                    'success': False,
                    'error': f"Student with ID {student_id} not found",
                    'student_id': student_id
                }
            
            update_result = {
                'success': True,
                'student_id': student_id,
                'name': student[1],
                'update_details': {}
            }
            
            # Add new image training
            if image_paths:
                image_result = self.face_trainer.train_from_multiple_images(image_paths, student_id)
                update_result['update_details']['image_training'] = image_result
            
            # Add new camera training
            if camera_samples > 0:
                camera_result = self.face_trainer.train_student_interactive(
                    student_id, num_samples=camera_samples
                )
                update_result['update_details']['camera_training'] = camera_result
            
            # Validate training quality
            validation_result = self.face_trainer.validate_training_quality(student_id)
            update_result['update_details']['validation'] = validation_result
            
            logger.info(f"Successfully updated training for student {student[1]} (ID: {student_id})")
            return update_result
        
        except Exception as e:
            logger.error(f"Error updating student training: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'student_id': student_id
            }
    
    def _validate_student_data(self, name: str, enrollment_number: str) -> Tuple[bool, str]:
        """Validate student registration data."""
        from utils.helpers import validate_student_data
        return validate_student_data(name, enrollment_number)


def create_training_manager(database_path: str = None) -> TrainingManager:
    """
    Factory function to create a training manager.
    
    Args:
        database_path (str, optional): Path to the database file
        
    Returns:
        TrainingManager: Configured training manager
    """
    try:
        db_path = database_path or Config.DATABASE_PATH
        db_manager = DatabaseManager(db_path)
        training_manager = TrainingManager(db_manager)
        
        logger.info("Created TrainingManager")
        return training_manager
    
    except Exception as e:
        logger.error(f"Error creating training manager: {str(e)}")
        raise