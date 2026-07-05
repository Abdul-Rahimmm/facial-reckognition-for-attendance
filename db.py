"""
SQLite storage for students, face embeddings, and attendance logs.
"""

import csv
import json
import logging
import os
import pickle
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from config import Config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Small thread-safe SQLite wrapper for local-first attendance data."""

    def __init__(self, db_path: str = "data/attendance.db"):
        self.db_path = db_path
        self.lock = threading.RLock()
        self._ensure_parent_dir()
        self.init_db()

    def _ensure_parent_dir(self) -> None:
        db_dir = os.path.dirname(self.db_path)
        if db_dir and self.db_path != ":memory:":
            os.makedirs(db_dir, exist_ok=True)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=Config.DATABASE_TIMEOUT)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            if self.db_path != ":memory:":
                conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            yield conn
        finally:
            conn.close()

    def init_db(self) -> None:
        with self.lock:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS students (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        enrollment_number TEXT UNIQUE NOT NULL,
                        category TEXT DEFAULT '',
                        image_paths TEXT DEFAULT '[]',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS face_embeddings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id INTEGER NOT NULL,
                        embedding BLOB NOT NULL,
                        encoding_format TEXT DEFAULT 'pickle',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS attendance_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id INTEGER NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        session TEXT DEFAULT 'default',
                        confidence REAL DEFAULT 0.0,
                        FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE
                    )
                    """
                )
                self._ensure_column(cursor, "face_embeddings", "encoding_format", "TEXT DEFAULT 'pickle'")
                self._ensure_column(cursor, "students", "category", "TEXT DEFAULT ''")
                self._ensure_column(cursor, "students", "image_paths", "TEXT DEFAULT '[]'")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_students_enrollment ON students(enrollment_number)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_students_category ON students(category)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_student ON face_embeddings(student_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance_logs(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_student_session ON attendance_logs(student_id, session, timestamp)")
                conn.commit()

    @staticmethod
    def _ensure_column(cursor: sqlite3.Cursor, table: str, column: str, definition: str) -> None:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = {row[1] for row in cursor.fetchall()}
        if column not in columns:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    @staticmethod
    def _clean_text(value: str, field: str, max_len: int = 120) -> str:
        cleaned = " ".join((value or "").strip().split())
        if not cleaned:
            raise ValueError(f"{field} cannot be empty")
        if len(cleaned) > max_len:
            raise ValueError(f"{field} is too long")
        return cleaned

    @staticmethod
    def _clean_optional_text(value: Optional[str], max_len: int = 120) -> str:
        cleaned = " ".join((value or "").strip().split())
        return cleaned[:max_len]

    @staticmethod
    def _validate_embedding(embedding: Sequence[Any]) -> List[float]:
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        values = [float(value) for value in embedding]
        if len(values) != 128:
            raise ValueError("embedding must contain 128 numeric values")
        return values

    @staticmethod
    def _serialize_embedding(embedding: Sequence[Any]) -> bytes:
        return json.dumps(DatabaseManager._validate_embedding(embedding), separators=(",", ":")).encode("utf-8")

    @staticmethod
    def _deserialize_embedding(data: bytes, encoding_format: Optional[str]) -> List[float]:
        if encoding_format == "json":
            return [float(value) for value in json.loads(data.decode("utf-8"))]
        # Backward compatibility for existing databases created by the original app.
        legacy = pickle.loads(data)
        if hasattr(legacy, "tolist"):
            legacy = legacy.tolist()
        return [float(value) for value in legacy]

    def add_student(
        self,
        name: str,
        enrollment_number: str,
        category: Optional[str] = None,
        image_paths: Optional[Sequence[str]] = None,
    ) -> int:
        name = self._clean_text(name, "name")
        enrollment_number = self._clean_text(enrollment_number, "enrollment number", 64)
        category = self._clean_optional_text(category, 80)
        serialized_paths = json.dumps([str(path) for path in (image_paths or [])])
        with self.lock:
            try:
                with self._connect() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO students (name, enrollment_number, category, image_paths) VALUES (?, ?, ?, ?)",
                        (name, enrollment_number, category, serialized_paths),
                    )
                    conn.commit()
                    return int(cursor.lastrowid)
            except sqlite3.IntegrityError as exc:
                if "UNIQUE constraint failed" in str(exc):
                    raise ValueError(f"Student with enrollment number {enrollment_number} already exists") from exc
                raise

    def get_student_by_id(self, student_id: int) -> Optional[Tuple]:
        with self.lock:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, name, enrollment_number, created_at FROM students WHERE id = ?", (student_id,))
                return cursor.fetchone()

    def get_student_by_enrollment(self, enrollment_number: str) -> Optional[Tuple]:
        with self.lock:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, name, enrollment_number, created_at FROM students WHERE enrollment_number = ?",
                    (enrollment_number,),
                )
                return cursor.fetchone()

    def get_all_students(self) -> List[Tuple]:
        with self.lock:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, name, enrollment_number FROM students ORDER BY name")
                students = cursor.fetchall()
        return [
            (student_id, name, enrollment_number, self.get_face_embeddings(student_id))
            for student_id, name, enrollment_number in students
        ]

    def list_registered_people(self) -> List[Dict[str, Any]]:
        with self.lock:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT
                        s.id,
                        s.name,
                        s.enrollment_number,
                        COALESCE(s.category, ''),
                        COALESCE(s.image_paths, '[]'),
                        s.created_at,
                        COUNT(fe.id)
                    FROM students s
                    LEFT JOIN face_embeddings fe ON fe.student_id = s.id
                    GROUP BY s.id
                    ORDER BY COALESCE(s.category, ''), s.name
                    """
                )
                rows = cursor.fetchall()

        people = []
        for student_id, name, enrollment_number, category, image_paths, created_at, embedding_count in rows:
            try:
                paths = json.loads(image_paths or "[]")
            except json.JSONDecodeError:
                paths = []
            people.append(
                {
                    "id": student_id,
                    "name": name,
                    "enrollment_number": enrollment_number,
                    "category": category,
                    "image_count": len(paths),
                    "embedding_count": embedding_count,
                    "created_at": created_at,
                }
            )
        return people

    def delete_student(self, student_id: int) -> bool:
        with self.lock:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM attendance_logs WHERE student_id = ?", (student_id,))
                cursor.execute("DELETE FROM face_embeddings WHERE student_id = ?", (student_id,))
                cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
                conn.commit()
                return cursor.rowcount > 0

    def add_face_embedding(self, student_id: int, embedding: Sequence[Any]) -> bool:
        try:
            serialized = self._serialize_embedding(embedding)
            with self.lock:
                with self._connect() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO face_embeddings (student_id, embedding, encoding_format) VALUES (?, ?, 'json')",
                        (student_id, sqlite3.Binary(serialized)),
                    )
                    conn.commit()
            return True
        except Exception as exc:
            logger.error("Failed to add face embedding: %s", exc)
            return False

    def get_face_embeddings(self, student_id: int) -> List[List[float]]:
        embeddings: List[List[float]] = []
        with self.lock:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT embedding, COALESCE(encoding_format, 'pickle') FROM face_embeddings WHERE student_id = ?",
                    (student_id,),
                )
                rows = cursor.fetchall()
        for data, encoding_format in rows:
            try:
                embeddings.append(self._deserialize_embedding(data, encoding_format))
            except Exception as exc:
                logger.warning("Skipping unreadable embedding for student %s: %s", student_id, exc)
        return embeddings

    def recently_logged(self, student_id: int, session: str, window_seconds: Optional[int] = None) -> bool:
        window_seconds = Config.DUPLICATE_LOG_WINDOW_SECONDS if window_seconds is None else window_seconds
        with self.lock:
            with self._connect() as conn:
                cursor = conn.cursor()
                if window_seconds is not None and window_seconds > 0:
                    cutoff = (datetime.now() - timedelta(seconds=window_seconds)).strftime("%Y-%m-%d %H:%M:%S")
                    cursor.execute(
                        """
                        SELECT 1 FROM attendance_logs
                        WHERE student_id = ? AND session = ? AND timestamp >= ?
                        LIMIT 1
                        """,
                        (student_id, session, cutoff),
                    )
                    if cursor.fetchone() is not None:
                        return True

                cursor.execute(
                    """
                    SELECT 1 FROM attendance_logs
                    WHERE student_id = ? AND session = ?
                    LIMIT 1
                    """,
                    (student_id, session),
                )
                return cursor.fetchone() is not None

    def log_attendance(
        self,
        student_id: int,
        session: str = "default",
        confidence: float = 0.0,
        suppress_duplicates: bool = True,
        duplicate_window_seconds: Optional[int] = None,
    ) -> bool:
        session = self._clean_text(session or "default", "session", 80)
        confidence = max(0.0, min(1.0, float(confidence)))
        if suppress_duplicates and self.recently_logged(student_id, session, duplicate_window_seconds):
            logger.debug("Skipped duplicate attendance for student %s in session %s", student_id, session)
            return False
        with self.lock:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO attendance_logs (student_id, session, confidence) VALUES (?, ?, ?)",
                    (student_id, session, confidence),
                )
                conn.commit()
        return True

    def get_attendance_logs(self, date: Optional[str] = None, session: Optional[str] = None) -> List[Tuple]:
        params: List[Any] = []
        filters: List[str] = []
        if date:
            datetime.strptime(date, "%Y-%m-%d")
            filters.append("DATE(al.timestamp) = ?")
            params.append(date)
        if session:
            filters.append("al.session = ?")
            params.append(session)
        where = " WHERE " + " AND ".join(filters) if filters else ""
        query = f"""
            SELECT al.student_id, s.name, al.timestamp, al.session, al.confidence
            FROM attendance_logs al
            JOIN students s ON al.student_id = s.id
            {where}
            ORDER BY al.timestamp DESC
        """
        with self.lock:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                return cursor.fetchall()

    def export_to_csv(self, filename: str, date: Optional[str] = None) -> bool:
        if not filename or not filename.strip():
            raise ValueError("CSV filename cannot be empty")
        target_dir = os.path.dirname(os.path.abspath(filename))
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        logs = self.get_attendance_logs(date=date)
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Student ID", "Name", "Timestamp", "Session", "Confidence"])
            writer.writerows(logs)
        return True

    def get_statistics(self) -> Dict[str, Any]:
        with self.lock:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM students")
                total_students = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM face_embeddings")
                total_embeddings = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM attendance_logs")
                total_attendance_records = cursor.fetchone()[0]
                cursor.execute(
                    """
                    SELECT AVG(embedding_count)
                    FROM (
                        SELECT student_id, COUNT(*) AS embedding_count
                        FROM face_embeddings
                        GROUP BY student_id
                    )
                    """
                )
                avg_embeddings = cursor.fetchone()[0] or 0
        return {
            "total_students": total_students,
            "total_embeddings": total_embeddings,
            "total_attendance_records": total_attendance_records,
            "average_embeddings_per_student": round(avg_embeddings, 2),
        }

    def close(self) -> None:
        logger.debug("DatabaseManager uses short-lived SQLite connections; nothing to close")
