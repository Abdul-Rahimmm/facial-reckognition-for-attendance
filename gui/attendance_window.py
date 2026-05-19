"""
Live attendance window for the Face Recognition Attendance System.

This module provides the GUI for live face recognition and attendance
marking from camera feed.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import threading
import time
from PIL import Image, ImageTk
import numpy as np
from datetime import datetime
import logging

from utils.logger import logger
from utils.config import Config
from utils.helpers import convert_to_rgb, get_face_locations, get_face_encodings
from recognition.face_detector import create_face_detector
from recognition.face_encoder import create_face_recognition_engine


class AttendanceWindow(tk.Frame):
    """Live attendance window class."""
    
    def __init__(self, parent: tk.Frame, attendance_system):
        """
        Initialize the attendance window.
        
        Args:
            parent (tk.Frame): Parent frame
            attendance_system: Main attendance system instance
        """
        super().__init__(parent, bg="white")
        self.attendance_system = attendance_system
        
        # Initialize variables
        self.camera_active = False
        self.camera_thread = None
        self.camera = None
        self.detector = None
        self.recognition_engine = None
        self.recognized_students = set()
        
        # Setup UI
        self.setup_ui()
        
        # Initialize components
        self.initialize_components()
        
        logger.info("Attendance window initialized")
    
    def setup_ui(self):
        """Setup the attendance window UI."""
        # Title
        title_label = tk.Label(
            self,
            text="Live Attendance",
            font=("Helvetica", 18, "bold"),
            bg="white",
            fg="#333333"
        )
        title_label.pack(pady=(0, 20))
        
        # Create main content frame
        content_frame = tk.Frame(self, bg="white")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Left panel: Camera feed
        left_panel = tk.LabelFrame(
            content_frame,
            text="Camera Feed",
            font=("Helvetica", 12, "bold"),
            bg="white",
            fg="#333333",
            padx=20,
            pady=20,
            width=500
        )
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        left_panel.pack_propagate(False)
        
        # Right panel: Controls and logs
        right_panel = tk.Frame(content_frame, bg="white")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Setup left panel (camera)
        self.setup_camera_panel(left_panel)
        
        # Setup right panel (controls and logs)
        self.setup_controls_panel(right_panel)
        self.setup_logs_panel(right_panel)
    
    def setup_camera_panel(self, parent: tk.LabelFrame):
        """Setup the camera panel."""
        # Camera frame
        camera_frame = tk.Frame(parent, bg="black", width=480, height=360)
        camera_frame.pack(pady=(0, 20))
        camera_frame.pack_propagate(False)
        
        # Camera display label
        self.camera_label = tk.Label(camera_frame, bg="black")
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Camera controls
        controls_frame = tk.Frame(parent, bg="white")
        controls_frame.pack(fill=tk.X)
        
        # Start/Stop camera button
        self.camera_btn = tk.Button(
            controls_frame,
            text="Start Camera",
            command=self.toggle_camera,
            font=("Helvetica", 10),
            bg="#3498db",
            fg="white",
            relief=tk.FLAT,
            padx=15,
            pady=5
        )
        self.camera_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Recognition status
        self.status_label = tk.Label(
            controls_frame,
            text="Status: Idle",
            font=("Helvetica", 10, "bold"),
            bg="white",
            fg="#333333"
        )
        self.status_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Recognition count
        self.count_label = tk.Label(
            controls_frame,
            text="Recognized: 0",
            font=("Helvetica", 10, "bold"),
            bg="white",
            fg="#333333"
        )
        self.count_label.pack(side=tk.LEFT)
    
    def setup_controls_panel(self, parent: tk.Frame):
        """Setup the controls panel."""
        controls_frame = tk.LabelFrame(
            parent,
            text="Controls",
            font=("Helvetica", 12, "bold"),
            bg="white",
            fg="#333333",
            padx=20,
            pady=20,
            height=150
        )
        controls_frame.pack(fill=tk.X, pady=(0, 20))
        controls_frame.pack_propagate(False)
        
        # Recognition threshold
        tk.Label(
            controls_frame,
            text="Recognition Threshold:",
            font=("Helvetica", 11),
            bg="white"
        ).pack(anchor=tk.W, pady=(0, 5))
        
        self.threshold_var = tk.DoubleVar(value=Config.RECOGNITION_THRESHOLD)
        threshold_slider = tk.Scale(
            controls_frame,
            from_=0.3,
            to=0.8,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self.threshold_var,
            command=self.update_threshold
        )
        threshold_slider.pack(fill=tk.X, pady=(0, 10))
        
        # Auto-clear option
        self.auto_clear_var = tk.BooleanVar(value=True)
        auto_clear_check = tk.Checkbutton(
            controls_frame,
            text="Auto-clear recognized students (5 minutes)",
            variable=self.auto_clear_var,
            font=("Helvetica", 10),
            bg="white"
        )
        auto_clear_check.pack(anchor=tk.W, pady=(0, 10))
        
        # Manual clear button
        clear_btn = tk.Button(
            controls_frame,
            text="Clear Recognized List",
            command=self.clear_recognized_list,
            font=("Helvetica", 10),
            bg="#e74c3c",
            fg="white",
            relief=tk.FLAT,
            padx=10,
            pady=5
        )
        clear_btn.pack(side=tk.LEFT)
    
    def setup_logs_panel(self, parent: tk.Frame):
        """Setup the logs panel."""
        logs_frame = tk.LabelFrame(
            parent,
            text="Attendance Logs",
            font=("Helvetica", 12, "bold"),
            bg="white",
            fg="#333333",
            padx=20,
            pady=20
        )
        logs_frame.pack(fill=tk.BOTH, expand=True)
        
        # Logs treeview
        columns = ("Time", "Student Name", "Enrollment", "Confidence")
        self.logs_tree = ttk.Treeview(
            logs_frame,
            columns=columns,
            show="headings",
            height=15
        )
        
        # Setup columns
        for col in columns:
            self.logs_tree.heading(col, text=col)
            self.logs_tree.column(col, width=120)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(logs_frame, orient=tk.VERTICAL, command=self.logs_tree.yview)
        self.logs_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack treeview and scrollbar
        self.logs_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def initialize_components(self):
        """Initialize face recognition components."""
        try:
            # Initialize face detector
            self.detector = create_face_detector()
            
            # Initialize recognition engine
            self.recognition_engine = create_face_recognition_engine()
            
            logger.info("Attendance components initialized")
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            messagebox.showerror("Initialization Error", f"Failed to initialize components: {str(e)}")
    
    def toggle_camera(self):
        """Toggle camera on/off."""
        if not self.camera_active:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start the camera feed and recognition."""
        try:
            self.camera = cv2.VideoCapture(Config.CAMERA_INDEX)
            if not self.camera.isOpened():
                messagebox.showerror("Error", "Cannot access camera")
                return
            
            self.camera_active = True
            self.camera_btn.config(text="Stop Camera", bg="#e74c3c")
            self.status_label.config(text="Status: Active", fg="green")
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.process_camera_feed)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
            logger.info("Camera and recognition started")
        
        except Exception as e:
            logger.error(f"Error starting camera: {str(e)}")
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
    
    def stop_camera(self):
        """Stop the camera feed and recognition."""
        try:
            self.camera_active = False
            if self.camera:
                self.camera.release()
            self.camera_btn.config(text="Start Camera", bg="#3498db")
            self.status_label.config(text="Status: Idle", fg="#333333")
            
            # Clear camera label
            self.camera_label.config(image="")
            
            logger.info("Camera and recognition stopped")
        
        except Exception as e:
            logger.error(f"Error stopping camera: {str(e)}")
    
    def process_camera_feed(self):
        """Process camera feed for face recognition."""
        try:
            last_clear_time = time.time()
            
            while self.camera_active:
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                face_locations = get_face_locations(rgb_frame)
                
                # Process faces if found
                if face_locations:
                    # Encode faces
                    face_encodings = get_face_encodings(rgb_frame, face_locations)
                    
                    if face_encodings:
                        # Get known faces from database
                        known_encodings, known_names, known_ids = self.get_known_faces()
                        
                        if known_encodings:
                            # Match faces
                            for i, face_encoding in enumerate(face_encodings):
                                match_result = self.match_face(
                                    face_encoding, known_encodings, known_names, known_ids
                                )
                                
                                if match_result['matched'] and match_result['student_id'] not in self.recognized_students:
                                    # Record attendance
                                    self.record_attendance(match_result)
                                    self.recognized_students.add(match_result['student_id'])
                                    self.update_count_label()
                
                # Draw face boxes
                if face_locations:
                    frame_with_boxes = self.draw_face_boxes(frame.copy(), face_locations)
                else:
                    frame_with_boxes = frame
                
                # Convert to display format
                display_frame = cv2.resize(frame_with_boxes, (480, 360))
                pil_image = Image.fromarray(display_frame)
                photo = ImageTk.PhotoImage(image=pil_image)
                
                # Update label
                self.camera_label.config(image=photo)
                self.camera_label.image = photo
                
                # Auto-clear recognized students
                if self.auto_clear_var.get():
                    current_time = time.time()
                    if current_time - last_clear_time > 300:  # 5 minutes
                        self.recognized_students.clear()
                        self.update_count_label()
                        last_clear_time = current_time
                
                time.sleep(0.1)  # Process at ~10 FPS
        
        except Exception as e:
            logger.error(f"Error processing camera feed: {str(e)}")
    
    def get_known_faces(self):
        """Get known faces from the recognition manager."""
        try:
            if self.attendance_system.recognition_manager:
                return self.attendance_system.recognition_manager.face_matcher.get_known_faces_from_database()
            return [], [], []
        except Exception as e:
            logger.error(f"Error getting known faces: {str(e)}")
            return [], [], []
    
    def match_face(self, face_encoding, known_encodings, known_names, known_ids):
        """Match a single face against known faces."""
        try:
            if not known_encodings or face_encoding is None:
                return {'matched': False, 'student_id': None, 'name': 'Unknown', 'confidence': 0.0}
            
            # Calculate distances
            distances = []
            for known_encoding in known_encodings:
                distance = np.linalg.norm(face_encoding - known_encoding)
                distances.append(distance)
            
            # Find closest match
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]
            confidence = max(0.0, 1.0 - (best_distance / self.threshold_var.get()))
            
            # Check if match is valid
            is_match = best_distance <= self.threshold_var.get()
            
            return {
                'matched': is_match,
                'student_id': known_ids[best_match_index] if is_match else None,
                'name': known_names[best_match_index] if is_match else 'Unknown',
                'confidence': confidence,
                'distance': best_distance
            }
        
        except Exception as e:
            logger.error(f"Error matching face: {str(e)}")
            return {'matched': False, 'student_id': None, 'name': 'Unknown', 'confidence': 0.0}
    
    def record_attendance(self, match_result):
        """Record attendance for a matched student."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Add to logs treeview
            self.logs_tree.insert(
                "",
                0,
                values=(
                    timestamp,
                    match_result['name'],
                    f"ID: {match_result['student_id']}",
                    f"{match_result['confidence']:.2f}"
                )
            )
            
            # Log the event
            logger.info(f"Attendance recorded: {match_result['name']} (ID: {match_result['student_id']}) "
                       f"with confidence {match_result['confidence']:.2f}")
        
        except Exception as e:
            logger.error(f"Error recording attendance: {str(e)}")
    
    def draw_face_boxes(self, frame, face_locations):
        """Draw bounding boxes around detected faces."""
        try:
            for (top, right, bottom, left) in face_locations:
                # Draw rectangle
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Add label
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, "Face", (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
            
            return frame
        
        except Exception as e:
            logger.error(f"Error drawing face boxes: {str(e)}")
            return frame
    
    def update_threshold(self, value):
        """Update the recognition threshold."""
        try:
            new_threshold = float(value)
            # Update config
            from utils.config import Config
            Config.update_threshold(new_threshold)
            
            logger.info(f"Recognition threshold updated to: {new_threshold}")
        
        except Exception as e:
            logger.error(f"Error updating threshold: {str(e)}")
    
    def update_count_label(self):
        """Update the recognized students count."""
        count = len(self.recognized_students)
        self.count_label.config(text=f"Recognized: {count}")
    
    def clear_recognized_list(self):
        """Clear the recognized students list."""
        try:
            self.recognized_students.clear()
            self.update_count_label()
            logger.info("Recognized students list cleared")
        
        except Exception as e:
            logger.error(f"Error clearing recognized list: {str(e)}")
    
    def destroy(self):
        """Clean up when destroying the window."""
        try:
            self.stop_camera()
            super().destroy()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")