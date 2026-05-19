"""
Database models for the Face Recognition Attendance System.

This module defines the SQLite database schema using SQLAlchemy ORM.
It includes models for Students, Attendance logs, and Face embeddings.
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Optional, Tuple
import numpy as np


class DatabaseManager:
    """Manages SQLite database operations for the attendance system."""
    
    def __init__(self, db_path: str = "data/attendance.db"):
        """
        Initialize the database manager.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self._ensure_db_directory()
        self._create_tables()
    
    def _ensure_db_directory(self):
        """Ensure the database directory exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create students table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    enrollment_number TEXT UNIQUE NOT NULL,
                    image_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create face embeddings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (student_id) REFERENCES students (id)
                )
            ''')
            
            # Create attendance logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL,
                    FOREIGN KEY (student_id) REFERENCES students (id)
                )
            ''')
            
            conn.commit()
    
    def add_student(self, name: str, enrollment_number: str, image_path: Optional[str] = None) -> int:
        """
        Add a new student to the database.
        
        Args:
            name (str): Student's full name
            enrollment_number (str): Unique enrollment number
            image_path (str, optional): Path to the student's image
            
        Returns:
            int: The ID of the newly created student
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO students (name, enrollment_number, image_path)
                VALUES (?, ?, ?)
            ''', (name, enrollment_number, image_path))
            conn.commit()
            return cursor.lastrowid
    
    def add_face_embedding(self, student_id: int, embedding: np.ndarray):
        """
        Add a face embedding for a student.
        
        Args:
            student_id (int): ID of the student
            embedding (np.ndarray): 128-dimensional face embedding
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Convert numpy array to bytes for storage
            embedding_bytes = embedding.tobytes()
            cursor.execute('''
                INSERT INTO face_embeddings (student_id, embedding)
                VALUES (?, ?)
            ''', (student_id, embedding_bytes))
            conn.commit()
    
    def add_attendance_log(self, student_id: int, confidence: float):
        """
        Add an attendance log entry.
        
        Args:
            student_id (int): ID of the student
            confidence (float): Recognition confidence score
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO attendance_logs (student_id, confidence)
                VALUES (?, ?)
            ''', (student_id, confidence))
            conn.commit()
    
    def get_student_by_id(self, student_id: int) -> Optional[Tuple]:
        """
        Get student information by ID.
        
        Args:
            student_id (int): Student ID
            
        Returns:
            Optional[Tuple]: Student data or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM students WHERE id = ?', (student_id,))
            return cursor.fetchone()
    
    def get_student_by_enrollment(self, enrollment_number: str) -> Optional[Tuple]:
        """
        Get student information by enrollment number.
        
        Args:
            enrollment_number (str): Student's enrollment number
            
        Returns:
            Optional[Tuple]: Student data or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM students WHERE enrollment_number = ?', (enrollment_number,))
            return cursor.fetchone()
    
    def get_all_students(self) -> List[Tuple]:
        """
        Get all students from the database.
        
        Returns:
            List[Tuple]: List of all student records
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM students ORDER BY name')
            return cursor.fetchall()
    
    def get_face_embeddings(self, student_id: int) -> List[np.ndarray]:
        """
        Get all face embeddings for a student.
        
        Args:
            student_id (int): Student ID
            
        Returns:
            List[np.ndarray]: List of face embeddings
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT embedding FROM face_embeddings WHERE student_id = ?', (student_id,))
            rows = cursor.fetchall()
            
            embeddings = []
            for row in rows:
                # Convert bytes back to numpy array
                embedding = np.frombuffer(row[0], dtype=np.float64)
                embeddings.append(embedding)
            
            return embeddings
    
    def get_all_embeddings(self) -> List[Tuple[int, np.ndarray]]:
        """
        Get all face embeddings from the database.
        
        Returns:
            List[Tuple[int, np.ndarray]]: List of (student_id, embedding) tuples
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT student_id, embedding FROM face_embeddings')
            rows = cursor.fetchall()
            
            embeddings = []
            for student_id, embedding_bytes in rows:
                # Convert bytes back to numpy array
                embedding = np.frombuffer(embedding_bytes, dtype=np.float64)
                embeddings.append((student_id, embedding))
            
            return embeddings
    
    def get_attendance_logs(self, student_id: Optional[int] = None, 
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> List[Tuple]:
        """
        Get attendance logs with optional filtering.
        
        Args:
            student_id (int, optional): Filter by student ID
            start_date (datetime, optional): Filter by start date
            end_date (datetime, optional): Filter by end date
            
        Returns:
            List[Tuple]: List of attendance log records
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = '''
                SELECT al.*, s.name, s.enrollment_number 
                FROM attendance_logs al
                JOIN students s ON al.student_id = s.id
            '''
            params = []
            
            if student_id is not None:
                query += ' WHERE al.student_id = ?'
                params.append(student_id)
            
            if start_date is not None:
                if 'WHERE' in query:
                    query += ' AND timestamp >= ?'
                else:
                    query += ' WHERE timestamp >= ?'
                params.append(start_date)
            
            if end_date is not None:
                if 'WHERE' in query:
                    query += ' AND timestamp <= ?'
                else:
                    query += ' WHERE timestamp <= ?'
                params.append(end_date)
            
            query += ' ORDER BY timestamp DESC'
            
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def delete_student(self, student_id: int):
        """
        Delete a student and all associated data.
        
        Args:
            student_id (int): ID of the student to delete
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Delete in correct order to avoid foreign key constraint issues
            cursor.execute('DELETE FROM attendance_logs WHERE student_id = ?', (student_id,))
            cursor.execute('DELETE FROM face_embeddings WHERE student_id = ?', (student_id,))
            cursor.execute('DELETE FROM students WHERE id = ?', (student_id,))
            conn.commit()
    
    def update_student(self, student_id: int, name: Optional[str] = None, 
                      enrollment_number: Optional[str] = None,
                      image_path: Optional[str] = None):
        """
        Update student information.
        
        Args:
            student_id (int): ID of the student to update
            name (str, optional): New name
            enrollment_number (str, optional): New enrollment number
            image_path (str, optional): New image path
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Build dynamic update query
            fields = []
            params = []
            
            if name is not None:
                fields.append('name = ?')
                params.append(name)
            
            if enrollment_number is not None:
                fields.append('enrollment_number = ?')
                params.append(enrollment_number)
            
            if image_path is not None:
                fields.append('image_path = ?')
                params.append(image_path)
            
            if fields:
                params.append(student_id)
                query = f'UPDATE students SET {", ".join(fields)} WHERE id = ?'
                cursor.execute(query, params)
                conn.commit()
    
    def get_student_count(self) -> int:
        """Get the total number of students in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM students')
            return cursor.fetchone()[0]
    
    def get_attendance_count(self, student_id: Optional[int] = None) -> int:
        """Get the total number of attendance records."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if student_id is not None:
                cursor.execute('SELECT COUNT(*) FROM attendance_logs WHERE student_id = ?', (student_id,))
            else:
                cursor.execute('SELECT COUNT(*) FROM attendance_logs')
            
            return cursor.fetchone()[0]