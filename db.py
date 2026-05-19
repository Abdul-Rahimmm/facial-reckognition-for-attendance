"""
Database module for the Face Recognition Attendance System.

This module provides SQLite database operations for student management,
face embeddings storage, and attendance logging with thread safety.
"""

import sqlite3
import pickle
import threading
import logging
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Thread-safe database manager for attendance system operations."""
    
    def __init__(self, db_path: str = "data/attendance.db"):
        """
        Initialize the database manager.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.lock = threading.RLock()  # Thread-safe reentrant lock
        self.init_db()
    
    def init_db(self):
        """Initialize database tables and schema."""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Create students table
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS students (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT NOT NULL,
                            enrollment_number TEXT UNIQUE NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                    
                    # Create face_embeddings table
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS face_embeddings (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            student_id INTEGER NOT NULL,
                            embedding BLOB NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (student_id) REFERENCES students (id)
                        )
                    ''')
                    
                    # Create attendance_logs table
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS attendance_logs (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            student_id INTEGER NOT NULL,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            session TEXT DEFAULT 'default',
                            confidence REAL DEFAULT 0.0,
                            FOREIGN KEY (student_id) REFERENCES students (id)
                        )
                    ''')
                    
                    conn.commit()
                    logger.info("Database initialized successfully")
        
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    def add_student(self, name: str, enrollment_number: str) -> int:
        """
        Add a new student to the database.
        
        Args:
            name (str): Student name
            enrollment_number (str): Unique enrollment number
            
        Returns:
            int: Student ID of the newly created student
            
        Raises:
            sqlite3.IntegrityError: If enrollment number already exists
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO students (name, enrollment_number) VALUES (?, ?)",
                        (name, enrollment_number)
                    )
                    conn.commit()
                    student_id = cursor.lastrowid
                    logger.info(f"Added student: {name} (ID: {student_id})")
                    return student_id
        
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                logger.error(f"Student with enrollment number {enrollment_number} already exists")
                raise Exception(f"Student with enrollment number {enrollment_number} already exists")
            else:
                logger.error(f"Database integrity error: {e}")
                raise
        
        except sqlite3.Error as e:
            logger.error(f"Database error adding student: {e}")
            raise
    
    def get_student_by_id(self, student_id: int) -> Optional[Tuple]:
        """
        Get student information by ID.
        
        Args:
            student_id (int): Student ID
            
        Returns:
            Optional[Tuple]: Student data (id, name, enrollment_number, created_at) or None
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT id, name, enrollment_number, created_at FROM students WHERE id = ?",
                        (student_id,)
                    )
                    return cursor.fetchone()
        
        except sqlite3.Error as e:
            logger.error(f"Database error getting student by ID: {e}")
            return None
    
    def get_student_by_enrollment(self, enrollment_number: str) -> Optional[Tuple]:
        """
        Get student information by enrollment number.
        
        Args:
            enrollment_number (str): Student enrollment number
            
        Returns:
            Optional[Tuple]: Student data (id, name, enrollment_number, created_at) or None
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT id, name, enrollment_number, created_at FROM students WHERE enrollment_number = ?",
                        (enrollment_number,)
                    )
                    return cursor.fetchone()
        
        except sqlite3.Error as e:
            logger.error(f"Database error getting student by enrollment: {e}")
            return None
    
    def get_all_students(self) -> List[Tuple]:
        """
        Get all students with their face embeddings.
        
        Returns:
            List[Tuple]: List of students with embeddings [(id, name, enrollment_number, embeddings), ...]
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Get all students
                    cursor.execute("SELECT id, name, enrollment_number FROM students")
                    students = cursor.fetchall()
                    
                    result = []
                    for student in students:
                        student_id, name, enrollment_number = student
                        
                        # Get embeddings for this student
                        embeddings = self.get_face_embeddings(student_id)
                        
                        result.append((student_id, name, enrollment_number, embeddings))
                    
                    logger.info(f"Loaded {len(result)} students with embeddings")
                    return result
        
        except sqlite3.Error as e:
            logger.error(f"Database error getting all students: {e}")
            return []
    
    def add_face_embedding(self, student_id: int, embedding: List[float]) -> bool:
        """
        Add a face embedding for a student.
        
        Args:
            student_id (int): Student ID
            embedding (List[float]): 128-dimensional face embedding
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate embedding
            if not isinstance(embedding, list) or len(embedding) != 128:
                logger.error("Invalid embedding format: must be a list of 128 floats")
                return False
            
            # Serialize embedding using pickle
            serialized_embedding = pickle.dumps(embedding)
            
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO face_embeddings (student_id, embedding) VALUES (?, ?)",
                        (student_id, sqlite3.Binary(serialized_embedding))
                    )
                    conn.commit()
                    logger.debug(f"Added face embedding for student ID: {student_id}")
                    return True
        
        except sqlite3.Error as e:
            logger.error(f"Database error adding face embedding: {e}")
            return False
    
    def get_face_embeddings(self, student_id: int) -> List[List[float]]:
        """
        Get all face embeddings for a student.
        
        Args:
            student_id (int): Student ID
            
        Returns:
            List[List[float]]: List of face embeddings
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT embedding FROM face_embeddings WHERE student_id = ?",
                        (student_id,)
                    )
                    rows = cursor.fetchall()
                    
                    embeddings = []
                    for row in rows:
                        # Deserialize embedding
                        embedding = pickle.loads(row[0])
                        embeddings.append(embedding)
                    
                    logger.debug(f"Retrieved {len(embeddings)} embeddings for student ID: {student_id}")
                    return embeddings
        
        except sqlite3.Error as e:
            logger.error(f"Database error getting face embeddings: {e}")
            return []
        except pickle.PickleError as e:
            logger.error(f"Error deserializing face embeddings: {e}")
            return []
    
    def log_attendance(self, student_id: int, session: str = "default", confidence: float = 0.0) -> bool:
        """
        Log attendance for a student.
        
        Args:
            student_id (int): Student ID
            session (str): Session identifier
            confidence (float): Recognition confidence score
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO attendance_logs (student_id, session, confidence) VALUES (?, ?, ?)",
                        (student_id, session, confidence)
                    )
                    conn.commit()
                    logger.info(f"Logged attendance for student ID: {student_id}, session: {session}, confidence: {confidence:.3f}")
                    return True
        
        except sqlite3.Error as e:
            logger.error(f"Database error logging attendance: {e}")
            return False
    
    def get_attendance_logs(self, date: Optional[str] = None, session: Optional[str] = None) -> List[Tuple]:
        """
        Get attendance logs with optional filtering.
        
        Args:
            date (Optional[str]): Date filter in 'YYYY-MM-DD' format
            session (Optional[str]): Session filter
            
        Returns:
            List[Tuple]: List of attendance logs [(student_id, name, timestamp, session, confidence), ...]
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Build query with optional filters
                    query = '''
                        SELECT al.student_id, s.name, al.timestamp, al.session, al.confidence
                        FROM attendance_logs al
                        JOIN students s ON al.student_id = s.id
                    '''
                    params = []
                    
                    if date:
                        query += " WHERE DATE(al.timestamp) = ?"
                        params.append(date)
                    
                    if session:
                        if date:
                            query += " AND al.session = ?"
                        else:
                            query += " WHERE al.session = ?"
                        params.append(session)
                    
                    query += " ORDER BY al.timestamp DESC"
                    
                    cursor.execute(query, params)
                    return cursor.fetchall()
        
        except sqlite3.Error as e:
            logger.error(f"Database error getting attendance logs: {e}")
            return []
    
    def export_to_csv(self, filename: str, date: Optional[str] = None) -> bool:
        """
        Export attendance logs to CSV file.
        
        Args:
            filename (str): Output CSV file path
            date (Optional[str]): Date filter in 'YYYY-MM-DD' format
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import csv
            
            logs = self.get_attendance_logs(date=date)
            
            if not logs:
                logger.warning("No attendance logs to export")
                return False
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(['Student ID', 'Name', 'Timestamp', 'Session', 'Confidence'])
                
                # Write data
                for log in logs:
                    writer.writerow(log)
            
            logger.info(f"Exported {len(logs)} attendance records to {filename}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dict[str, Any]: Database statistics
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Count students
                    cursor.execute("SELECT COUNT(*) FROM students")
                    total_students = cursor.fetchone()[0]
                    
                    # Count embeddings
                    cursor.execute("SELECT COUNT(*) FROM face_embeddings")
                    total_embeddings = cursor.fetchone()[0]
                    
                    # Count attendance records
                    cursor.execute("SELECT COUNT(*) FROM attendance_logs")
                    total_attendance_records = cursor.fetchone()[0]
                    
                    # Average embeddings per student
                    if total_students > 0:
                        cursor.execute("""
                            SELECT AVG(embedding_count) 
                            FROM (
                                SELECT student_id, COUNT(*) as embedding_count 
                                FROM face_embeddings 
                                GROUP BY student_id
                            )
                        """)
                        avg_embeddings_per_student = cursor.fetchone()[0] or 0
                    else:
                        avg_embeddings_per_student = 0
                    
                    return {
                        'total_students': total_students,
                        'total_embeddings': total_embeddings,
                        'total_attendance_records': total_attendance_records,
                        'average_embeddings_per_student': round(avg_embeddings_per_student, 2)
                    }
        
        except sqlite3.Error as e:
            logger.error(f"Database error getting statistics: {e}")
            return {
                'total_students': 0,
                'total_embeddings': 0,
                'total_attendance_records': 0,
                'average_embeddings_per_student': 0
            }
    
    def close(self):
        """Close database connection."""
        # SQLite connections are automatically managed, but this provides explicit cleanup
        logger.info("Database connection closed")