"""
Main application entry point for the Face Recognition Attendance System.

This module integrates the database, recognition, and GUI modules
to create a complete face recognition attendance system.
"""

import sys
import os
import logging
import argparse
import threading
import time
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from config import Config, ErrorHandler
from db import DatabaseManager
from recognition import FaceRecognitionSystem, capture_images, detect_faces, extract_embeddings, recognize_and_log
from gui import AttendanceGUI
import cv2
import tkinter as tk

# Configure logging with enhanced configuration
Config.ensure_directories()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AttendanceSystem:
    """Main attendance system class that integrates all modules."""
    
    def __init__(self, database_path: str = "data/attendance.db"):
        """
        Initialize the attendance system.
        
        Args:
            database_path (str): Path to the database file
        """
        self.database_path = database_path
        self.db_manager = None
        self.face_system = None
        self.known_students = []
        self.is_initialized = False
        
        logger.info("Initializing Face Recognition Attendance System")
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            # Initialize database
            logger.info("Initializing database...")
            self.db_manager = DatabaseManager(self.database_path)
            
            # Initialize face recognition system
            logger.info("Initializing face recognition system...")
            self.face_system = FaceRecognitionSystem(tolerance=Config.RECOGNITION_TOLERANCE, camera_index=Config.CAMERA_INDEX)
            
            # Load known students
            self.load_known_students()
            
            self.is_initialized = True
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            self.is_initialized = False
            raise
    
    def load_known_students(self):
        """Load known students from database."""
        try:
            self.known_students = self.db_manager.get_all_students()
            logger.info(f"Loaded {len(self.known_students)} known students")
            
            # Update face system with known students
            self.face_system.update_known_students(self.known_students)
            
        except Exception as e:
            logger.error(f"Error loading known students: {str(e)}")
            self.known_students = []
    
    def get_system_status(self) -> dict:
        """
        Get overall system status.
        
        Returns:
            dict: System status information
        """
        try:
            stats = self.db_manager.get_statistics()
            
            status = {
                'initialized': self.is_initialized,
                'database_connected': self.db_manager is not None,
                'face_recognition_ready': self.face_system is not None,
                'known_students': len(self.known_students),
                'database_path': self.database_path,
                'tolerance': self.face_system.tolerance if self.face_system else Config.RECOGNITION_TOLERANCE,
                'database_stats': stats
            }
            
            return status
        
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {'error': str(e), 'initialized': False}
    
    def register_student(self, name: str, enrollment_number: str = None, image_paths: list = None, num_images: int = 5) -> dict:
        """
        Register a new student with face training.
        
        Args:
            name (str): Student name
            num_images (int): Number of images to capture
            
        Returns:
            dict: Registration result
        """
        try:
            if not self.is_initialized:
                return {'success': False, 'error': 'System not initialized'}
            
            logger.info(f"Starting registration for student: {name}")

            # Backwards compatibility: if caller passed (name, num_images)
            if isinstance(enrollment_number, int):
                num_images = enrollment_number
                enrollment_number = None

            # Determine enrollment number
            if not enrollment_number:
                enrollment_number = f"STU{int(time.time())}"

            # Prepare embeddings list
            embeddings = []

            # If image paths provided, extract embeddings from them
            if image_paths:
                for img_path in image_paths:
                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            logger.warning(f"Failed to read image: {img_path}")
                            continue

                        # Detect faces and extract embeddings
                        face_locs = detect_faces(img)
                        if not face_locs:
                            logger.warning(f"No faces detected in image: {img_path}")
                            continue

                        encs = extract_embeddings(img, face_locs)
                        if not encs:
                            logger.warning(f"No face encodings extracted from: {img_path}")
                            continue

                        # If multiple faces found, take first encoding
                        embeddings.append(encs[0])

                    except Exception as e:
                        logger.error(f"Error processing image {img_path}: {e}")

            # If no embeddings from images, fall back to camera capture
            if not embeddings:
                logger.info("No valid embeddings from uploaded images, attempting camera capture")
                embeddings = capture_images(name, num_images)

            if not embeddings:
                return {'success': False, 'error': 'No images captured or processed successfully'}

            # Add student to database
            student_id = self.db_manager.add_student(name, enrollment_number)

            # Add embeddings to database
            for embedding in embeddings:
                self.db_manager.add_face_embedding(student_id, embedding)
            
            # Reload known students
            self.load_known_students()
            
            result = {
                'success': True,
                'student_id': student_id,
                'name': name,
                'enrollment_number': enrollment_number,
                'embeddings_created': len(embeddings)
            }
            
            logger.info(f"Successfully registered student: {name} (ID: {student_id})")
            return result
        
        except Exception as e:
            logger.error(f"Error registering student: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def start_attendance_loop(self, session: str = 'default', callback=None):
        """
        Start the attendance taking loop with live video processing.
        
        Args:
            session (str): Session identifier for attendance logging
            callback (function): Optional callback function for frame updates
        """
        try:
            if not self.is_initialized:
                logger.error("System not initialized")
                return False
            
            if not self.known_students:
                logger.warning("No known students found")
                return False
            
            # Initialize camera
            camera = cv2.VideoCapture(Config.CAMERA_INDEX)
            if not camera.isOpened():
                logger.error("Cannot access camera")
                return False
            
            logger.info(f"Starting attendance loop for session: {session}")
            
            frame_count = 0
            last_update = time.time()
            
            while True:
                ret, frame = camera.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break
                
                frame_count += 1
                
                # Process frame for recognition
                processed_frame = recognize_and_log(
                    frame, 
                    self.known_students, 
                    self.db_manager, 
                    session=session, 
                    tolerance=self.face_system.tolerance
                )
                
                # Call callback with processed frame if provided
                if callback:
                    callback(processed_frame)
                
                # Update every 30 frames (~1 second at 30 FPS)
                if frame_count % 30 == 0:
                    current_time = time.time()
                    fps = 30 / (current_time - last_update)
                    last_update = current_time
                    
                    # Get today's attendance count
                    today_logs = self.db_manager.get_attendance_logs(date=datetime.now().strftime("%Y-%m-%d"))
                    
                    logger.info(f"Processing... FPS: {fps:.1f}, Today's Attendance: {len(today_logs)}")
                
                # Break loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            camera.release()
            cv2.destroyAllWindows()
            logger.info("Attendance loop stopped")
            return True
        
        except Exception as e:
            logger.error(f"Error in attendance loop: {str(e)}")
            return False
    
    def get_attendance_logs(self, date: str = None) -> list:
        """
        Get attendance logs.
        
        Args:
            date (str, optional): Date filter in 'YYYY-MM-DD' format
            
        Returns:
            list: List of attendance logs
        """
        try:
            return self.db_manager.get_attendance_logs(date=date)
        except Exception as e:
            logger.error(f"Error getting attendance logs: {str(e)}")
            return []
    
    def export_logs_to_csv(self, filename: str, date: str = None) -> bool:
        """
        Export logs to CSV file.
        
        Args:
            filename (str): Output CSV file path
            date (str, optional): Date filter in 'YYYY-MM-DD' format
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            return self.db_manager.export_to_csv(filename, date=date)
        except Exception as e:
            logger.error(f"Error exporting logs: {str(e)}")
            return False
    
    def cleanup(self):
        """Clean up system resources."""
        try:
            logger.info("Cleaning up system resources...")
            # Database cleanup is handled automatically
            logger.info("System cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


def run_gui_application():
    """Run the GUI application."""
    try:
        # Create main application
        app = AttendanceSystem()
        
        # Check if system is properly initialized
        status = app.get_system_status()
        if not status.get('initialized', False):
            logger.error("System failed to initialize. Cannot start GUI.")
            print("System initialization failed. Cannot start GUI.")
            return
        
        # Create and run GUI
        root = tk.Tk()
        gui_app = AttendanceGUI(root)
        gui_app.set_system(app)  # Inject system instance into GUI
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Error running GUI application: {str(e)}")
        print(f"Failed to start GUI application: {str(e)}")
        print("Please check the logs for more details.")


def run_command_line_interface():
    """Run the command line interface."""
    try:
        print("Face Recognition Attendance System - Command Line Interface")
        print("=" * 60)
        
        # Create main application
        app = AttendanceSystem()
        
        # Check system status
        status = app.get_system_status()
        print(f"System Status: {'✓ Ready' if status.get('initialized') else '✗ Not Ready'}")
        
        if not status.get('initialized'):
            print("System initialization failed. Please check the logs.")
            return
        
        # Show basic statistics
        stats = status.get('database_stats', {})
        print(f"Total Students: {stats.get('total_students', 0)}")
        print(f"Total Embeddings: {stats.get('total_embeddings', 0)}")
        print(f"Total Attendance Records: {stats.get('total_attendance_records', 0)}")
        
        print("\nAvailable commands:")
        print("1. register - Register a new student")
        print("2. attendance - Start attendance taking (live video)")
        print("3. logs - View attendance logs")
        print("4. export - Export logs to CSV")
        print("5. quit - Exit")
        
        while True:
            command = input("\nEnter command: ").strip().lower()
            
            if command == 'quit' or command == 'exit':
                break
            elif command == 'register':
                name = input("Enter student name: ").strip()
                if name:
                    num_images = input("Enter number of images to capture (default 5): ").strip()
                    num_images = int(num_images) if num_images.isdigit() else 5
                    result = app.register_student(name, num_images)
                    if result['success']:
                        print(f"✓ Successfully registered {name}")
                    else:
                        print(f"✗ Registration failed: {result['error']}")
                else:
                    print("✗ Name cannot be empty")
            
            elif command == 'attendance':
                session = input("Enter session name (default 'default'): ").strip() or 'default'
                print("Starting attendance taking... Press 'q' to stop")
                app.start_attendance_loop(session)
            
            elif command == 'logs':
                date = input("Enter date (YYYY-MM-DD) or press Enter for all: ").strip()
                date = date if date else None
                logs = app.get_attendance_logs(date)
                if logs:
                    print(f"\nFound {len(logs)} attendance records:")
                    for log in logs[:10]:  # Show first 10
                        print(f"  {log[2]} | {log[1]} (ID: {log[0]}) | Session: {log[3]} | Confidence: {log[4]:.2f}")
                    if len(logs) > 10:
                        print(f"  ... and {len(logs) - 10} more")
                else:
                    print("No logs found")
            
            elif command == 'export':
                filename = input("Enter CSV filename: ").strip()
                if filename:
                    date = input("Enter date filter (YYYY-MM-DD) or press Enter for all: ").strip()
                    date = date if date else None
                    success = app.export_logs_to_csv(filename, date)
                    if success:
                        print(f"✓ Logs exported to {filename}")
                    else:
                        print("✗ Export failed")
                else:
                    print("✗ Filename cannot be empty")
            
            else:
                print("✗ Unknown command")
        
        print("Goodbye!")
        
    except Exception as e:
        logger.error(f"Error running CLI: {str(e)}")
        print(f"Failed to run CLI: {str(e)}")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='Face Recognition Attendance System')
    parser.add_argument('--mode', choices=['gui', 'cli'], default='gui',
                       help='Run mode: gui (default) or cli')
    parser.add_argument('--database', type=str, default="data/attendance.db",
                       help='Path to database file')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.mode == 'gui':
            run_gui_application()
        elif args.mode == 'cli':
            run_command_line_interface()
    
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"Application failed: {str(e)}")
        print("Please check the logs for more details.")
    
    finally:
        # Cleanup
        try:
            # Any global cleanup can be done here
            pass
        except Exception as e:
            logger.error(f"Error during final cleanup: {str(e)}")


if __name__ == "__main__":
    main()