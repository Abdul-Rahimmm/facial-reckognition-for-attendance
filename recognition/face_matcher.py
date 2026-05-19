"""
Face matching module for the Face Recognition Attendance System.

This module handles face matching and recognition against the database
with configurable thresholds and confidence scoring.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

from utils.logger import logger
from utils.config import Config
from database.models import DatabaseManager


class FaceMatcher:
    """Advanced face matching with database integration and confidence scoring."""
    
    def __init__(self, database_manager: DatabaseManager, tolerance: float = None):
        """
        Initialize the face matcher.
        
        Args:
            database_manager (DatabaseManager): Database manager instance
            tolerance (float, optional): Matching tolerance threshold
        """
        self.db_manager = database_manager
        self.tolerance = tolerance or Config.RECOGNITION_THRESHOLD
        
        logger.info(f"FaceMatcher initialized with tolerance: {self.tolerance}")
    
    def get_known_faces_from_database(self) -> Tuple[List[np.ndarray], List[str], List[int]]:
        """
        Retrieve all known face encodings and names from database.
        
        Returns:
            Tuple[List[np.ndarray], List[str], List[int]]: 
                (encodings, names, student_ids)
        """
        try:
            # Get all embeddings from database
            embeddings_data = self.db_manager.get_all_embeddings()
            
            if not embeddings_data:
                logger.warning("No face embeddings found in database")
                return [], [], []
            
            # Separate embeddings and student IDs
            encodings = []
            student_ids = []
            
            for student_id, embedding in embeddings_data:
                encodings.append(embedding)
                student_ids.append(student_id)
            
            # Get student names
            names = []
            for student_id in student_ids:
                student_data = self.db_manager.get_student_by_id(student_id)
                if student_data:
                    names.append(student_data[1])  # name is at index 1
                else:
                    names.append(f"Unknown_{student_id}")
            
            logger.info(f"Loaded {len(encodings)} face encodings from database")
            return encodings, names, student_ids
        
        except Exception as e:
            logger.error(f"Error loading known faces from database: {str(e)}")
            return [], [], []
    
    def match_face(self, face_encoding: np.ndarray, known_encodings: List[np.ndarray], 
                   known_names: List[str], known_ids: List[int]) -> Dict:
        """
        Match a single face encoding against known faces.
        
        Args:
            face_encoding (np.ndarray): Face encoding to match
            known_encodings (List[np.ndarray]): List of known face encodings
            known_names (List[str]): List of corresponding names
            known_ids (List[int]): List of corresponding student IDs
            
        Returns:
            Dict: Match result with confidence and metadata
        """
        try:
            if not known_encodings or face_encoding is None:
                return {
                    'matched': False,
                    'student_id': None,
                    'name': 'Unknown',
                    'confidence': 0.0,
                    'distance': float('inf'),
                    'threshold': self.tolerance
                }
            
            # Calculate distances to all known faces
            distances = []
            for known_encoding in known_encodings:
                distance = np.linalg.norm(face_encoding - known_encoding)
                distances.append(distance)
            
            # Find the closest match
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]
            
            # Calculate confidence (convert distance to confidence score)
            # Distance of 0 = 100% confidence, distance of tolerance = 0% confidence
            confidence = max(0.0, 1.0 - (best_distance / self.tolerance))
            
            # Check if match is valid
            is_match = best_distance <= self.tolerance
            
            result = {
                'matched': is_match,
                'student_id': known_ids[best_match_index] if is_match else None,
                'name': known_names[best_match_index] if is_match else 'Unknown',
                'confidence': confidence,
                'distance': best_distance,
                'threshold': self.tolerance,
                'best_match_index': best_match_index if is_match else None
            }
            
            logger.debug(f"Face match result: {result}")
            return result
        
        except Exception as e:
            logger.error(f"Error matching face: {str(e)}")
            return {
                'matched': False,
                'student_id': None,
                'name': 'Unknown',
                'confidence': 0.0,
                'distance': float('inf'),
                'threshold': self.tolerance,
                'error': str(e)
            }
    
    def match_faces_batch(self, face_encodings: List[np.ndarray]) -> List[Dict]:
        """
        Match multiple face encodings against database.
        
        Args:
            face_encodings (List[np.ndarray]): List of face encodings to match
            
        Returns:
            List[Dict]: List of match results
        """
        try:
            # Load known faces from database
            known_encodings, known_names, known_ids = self.get_known_faces_from_database()
            
            if not known_encodings:
                # Return unknown results for all encodings
                return [{
                    'matched': False,
                    'student_id': None,
                    'name': 'Unknown',
                    'confidence': 0.0,
                    'distance': float('inf'),
                    'threshold': self.tolerance
                } for _ in face_encodings]
            
            # Match each face encoding
            results = []
            for face_encoding in face_encodings:
                match_result = self.match_face(face_encoding, known_encodings, known_names, known_ids)
                results.append(match_result)
            
            logger.info(f"Matched {len(results)} face encodings")
            return results
        
        except Exception as e:
            logger.error(f"Error matching faces batch: {str(e)}")
            return []
    
    def register_new_face(self, student_id: int, face_encoding: np.ndarray) -> bool:
        """
        Register a new face encoding for a student.
        
        Args:
            student_id (int): ID of the student
            face_encoding (np.ndarray): Face encoding to register
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate student exists
            student = self.db_manager.get_student_by_id(student_id)
            if not student:
                logger.error(f"Student with ID {student_id} not found")
                return False
            
            # Add face embedding to database
            self.db_manager.add_face_embedding(student_id, face_encoding)
            
            logger.info(f"Registered new face encoding for student {student_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error registering new face: {str(e)}")
            return False
    
    def update_face_encoding(self, student_id: int, new_encoding: np.ndarray) -> bool:
        """
        Update existing face encoding for a student.
        
        Args:
            student_id (int): ID of the student
            new_encoding (np.ndarray): New face encoding
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate student exists
            student = self.db_manager.get_student_by_id(student_id)
            if not student:
                logger.error(f"Student with ID {student_id} not found")
                return False
            
            # For now, we'll add a new encoding rather than replacing
            # This allows for multiple reference images per student
            self.db_manager.add_face_embedding(student_id, new_encoding)
            
            logger.info(f"Updated face encoding for student {student_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error updating face encoding: {str(e)}")
            return False
    
    def get_student_match_history(self, student_id: int, limit: int = 10) -> List[Dict]:
        """
        Get match history for a specific student.
        
        Args:
            student_id (int): ID of the student
            limit (int): Maximum number of records to return
            
        Returns:
            List[Dict]: List of match history records
        """
        try:
            # Get attendance logs for the student
            logs = self.db_manager.get_attendance_logs(student_id=student_id)
            
            history = []
            for log in logs[:limit]:
                history.append({
                    'timestamp': log[2],  # timestamp
                    'confidence': log[3],  # confidence
                    'student_name': log[4],  # name from joined query
                    'enrollment_number': log[5]  # enrollment number from joined query
                })
            
            return history
        
        except Exception as e:
            logger.error(f"Error getting student match history: {str(e)}")
            return []
    
    def get_recognition_statistics(self) -> Dict:
        """
        Get recognition statistics and performance metrics.
        
        Returns:
            Dict: Statistics about the recognition system
        """
        try:
            # Get database statistics
            student_count = self.db_manager.get_student_count()
            attendance_count = self.db_manager.get_attendance_count()
            
            # Get known faces from database
            known_encodings, known_names, known_ids = self.get_known_faces_from_database()
            
            stats = {
                'total_students': student_count,
                'total_attendance_records': attendance_count,
                'total_face_encodings': len(known_encodings),
                'average_encodings_per_student': len(known_encodings) / student_count if student_count > 0 else 0,
                'matching_tolerance': self.tolerance,
                'tolerance_range': Config.TOLERANCE_RANGE,
                'database_path': self.db_manager.db_path
            }
            
            logger.info(f"Recognition statistics: {stats}")
            return stats
        
        except Exception as e:
            logger.error(f"Error getting recognition statistics: {str(e)}")
            return {'error': str(e)}
    
    def set_tolerance(self, tolerance: float):
        """
        Set the matching tolerance.
        
        Args:
            tolerance (float): New tolerance value
        """
        if Config.validate_threshold(tolerance):
            self.tolerance = tolerance
            logger.info(f"Matching tolerance updated to: {self.tolerance}")
        else:
            logger.warning(f"Invalid tolerance value: {tolerance}. Current tolerance: {self.tolerance}")
    
    def validate_face_encoding(self, encoding: np.ndarray) -> Dict:
        """
        Validate a face encoding for quality and format.
        
        Args:
            encoding (np.ndarray): Face encoding to validate
            
        Returns:
            Dict: Validation results
        """
        try:
            if encoding is None:
                return {'valid': False, 'error': 'Encoding is None'}
            
            if not isinstance(encoding, np.ndarray):
                return {'valid': False, 'error': 'Encoding is not a numpy array'}
            
            if encoding.shape != (128,):
                return {'valid': False, 'error': f'Invalid encoding shape: {encoding.shape}, expected (128,)'}
            
            # Check for reasonable values
            if np.isnan(encoding).any() or np.isinf(encoding).any():
                return {'valid': False, 'error': 'Encoding contains NaN or Inf values'}
            
            # Check magnitude
            magnitude = np.linalg.norm(encoding)
            if magnitude < 0.1 or magnitude > 10.0:
                return {'valid': False, 'error': f'Encoding magnitude out of range: {magnitude}'}
            
            return {
                'valid': True,
                'dimension': len(encoding),
                'magnitude': float(magnitude),
                'mean': float(np.mean(encoding)),
                'std': float(np.std(encoding))
            }
        
        except Exception as e:
            return {'valid': False, 'error': str(e)}


class RecognitionManager:
    """High-level manager for face recognition operations."""
    
    def __init__(self, database_manager: DatabaseManager):
        """
        Initialize the recognition manager.
        
        Args:
            database_manager (DatabaseManager): Database manager instance
        """
        self.db_manager = database_manager
        self.face_matcher = FaceMatcher(database_manager)
        
        logger.info("RecognitionManager initialized")
    
    def process_attendance(self, face_encodings: List[np.ndarray], frame_timestamp: str = None) -> List[Dict]:
        """
        Process attendance for multiple face encodings.
        
        Args:
            face_encodings (List[np.ndarray]): List of face encodings from current frame
            frame_timestamp (str, optional): Timestamp for the frame
            
        Returns:
            List[Dict]: List of attendance processing results
        """
        try:
            # Match faces against database
            match_results = self.face_matcher.match_faces_batch(face_encodings)
            
            processed_results = []
            
            for i, match_result in enumerate(match_results):
                result = {
                    'face_index': i,
                    'matched': match_result['matched'],
                    'student_id': match_result['student_id'],
                    'name': match_result['name'],
                    'confidence': match_result['confidence'],
                    'distance': match_result['distance'],
                    'timestamp': frame_timestamp,
                    'attendance_recorded': False
                }
                
                # Record attendance if match was successful
                if match_result['matched'] and match_result['confidence'] > 0.5:
                    try:
                        self.db_manager.add_attendance_log(
                            match_result['student_id'], 
                            match_result['confidence']
                        )
                        result['attendance_recorded'] = True
                        logger.info(f"Attendance recorded for {match_result['name']} with confidence {match_result['confidence']:.2f}")
                    except Exception as e:
                        logger.error(f"Failed to record attendance for {match_result['name']}: {str(e)}")
                
                processed_results.append(result)
            
            return processed_results
        
        except Exception as e:
            logger.error(f"Error processing attendance: {str(e)}")
            return []
    
    def register_student_face(self, student_id: int, face_encoding: np.ndarray, 
                            image_path: str = None) -> bool:
        """
        Register a student's face for attendance.
        
        Args:
            student_id (int): ID of the student
            face_encoding (np.ndarray): Face encoding to register
            image_path (str, optional): Path to the student's image
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate inputs
            validation = self.face_matcher.validate_face_encoding(face_encoding)
            if not validation['valid']:
                logger.error(f"Invalid face encoding: {validation['error']}")
                return False
            
            # Register the face encoding
            success = self.face_matcher.register_new_face(student_id, face_encoding)
            
            if success and image_path:
                # Update student record with image path if provided
                student = self.db_manager.get_student_by_id(student_id)
                if student and not student[3]:  # image_path is at index 3
                    self.db_manager.update_student(student_id, image_path=image_path)
            
            return success
        
        except Exception as e:
            logger.error(f"Error registering student face: {str(e)}")
            return False
    
    def get_system_status(self) -> Dict:
        """
        Get overall system status and statistics.
        
        Returns:
            Dict: System status information
        """
        try:
            stats = self.face_matcher.get_recognition_statistics()
            
            status = {
                'system_status': 'operational',
                'database_connected': True,
                'face_matcher_configured': True,
                'statistics': stats,
                'last_updated': '2024-01-01 00:00:00'  # Would be dynamic in real implementation
            }
            
            return status
        
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {
                'system_status': 'error',
                'error': str(e),
                'database_connected': False,
                'face_matcher_configured': False
            }


def create_recognition_manager(database_path: str = None) -> RecognitionManager:
    """
    Factory function to create a recognition manager.
    
    Args:
        database_path (str, optional): Path to the database file
        
    Returns:
        RecognitionManager: Configured recognition manager
    """
    try:
        db_path = database_path or Config.DATABASE_PATH
        db_manager = DatabaseManager(db_path)
        recognition_manager = RecognitionManager(db_manager)
        
        logger.info("Created RecognitionManager")
        return recognition_manager
    
    except Exception as e:
        logger.error(f"Error creating recognition manager: {str(e)}")
        raise