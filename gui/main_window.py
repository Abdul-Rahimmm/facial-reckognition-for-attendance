"""
Main GUI window for the Face Recognition Attendance System.

This module provides the main application window with navigation
between different modules (registration, attendance, reports).
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import threading
import time
from PIL import Image, ImageTk
import logging
from typing import Optional

from utils.logger import logger
from utils.config import Config
from recognition.face_detector import create_face_detector
from recognition.face_encoder import create_face_recognition_engine
from recognition.trainer import create_training_manager
from recognition.face_matcher import create_recognition_manager


class MainWindow:
    """Main application window class."""
    
    def __init__(self, root: tk.Tk, attendance_system):
        """
        Initialize the main window.
        
        Args:
            root (tk.Tk): Root tkinter window
            attendance_system: Main attendance system instance
        """
        self.root = root
        self.attendance_system = attendance_system
        
        # Window configuration
        self.root.title(Config.WINDOW_TITLE)
        self.root.geometry(f"{Config.WINDOW_WIDTH}x{Config.WINDOW_HEIGHT}")
        self.root.configure(bg=Config.THEME_COLOR)
        
        # Center the window
        self.center_window()
        
        # Initialize variables
        self.current_frame = None
        self.camera_active = False
        self.camera_thread = None
        self.detector = None
        self.recognition_engine = None
        
        # Setup UI
        self.setup_ui()
        
        # Initialize components
        self.initialize_components()
        
        logger.info("Main window initialized")
    
    def center_window(self):
        """Center the window on the screen."""
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (Config.WINDOW_WIDTH // 2)
        y = (self.root.winfo_screenheight() // 2) - (Config.WINDOW_HEIGHT // 2)
        self.root.geometry(f"+{x}+{y}")
    
    def setup_ui(self):
        """Setup the main window UI."""
        # Create main container
        self.main_container = tk.Frame(self.root, bg=Config.THEME_COLOR)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create header
        self.create_header()
        
        # Create navigation menu
        self.create_navigation()
        
        # Create content area
        self.create_content_area()
        
        # Create status bar
        self.create_status_bar()
    
    def create_header(self):
        """Create the application header."""
        header_frame = tk.Frame(self.main_container, bg=Config.THEME_COLOR)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Title
        title_label = tk.Label(
            header_frame, 
            text="Face Recognition Attendance System",
            font=("Helvetica", 24, "bold"),
            fg="white",
            bg=Config.THEME_COLOR
        )
        title_label.pack(side=tk.LEFT)
        
        # System status indicator
        self.status_indicator = tk.Label(
            header_frame,
            text="●",
            font=("Helvetica", 16),
            fg="red",
            bg=Config.THEME_COLOR
        )
        self.status_indicator.pack(side=tk.RIGHT, padx=10)
        
        status_label = tk.Label(
            header_frame,
            text="System Status: Checking...",
            font=("Helvetica", 10),
            fg="white",
            bg=Config.THEME_COLOR
        )
        status_label.pack(side=tk.RIGHT)
    
    def create_navigation(self):
        """Create the navigation menu."""
        nav_frame = tk.Frame(self.main_container, bg=Config.THEME_COLOR, width=200)
        nav_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        nav_frame.pack_propagate(False)
        
        # Navigation buttons
        nav_buttons = [
            ("Dashboard", self.show_dashboard),
            ("Register Student", self.show_registration),
            ("Live Attendance", self.show_attendance),
            ("Reports", self.show_reports),
            ("Settings", self.show_settings)
        ]
        
        for text, command in nav_buttons:
            btn = tk.Button(
                nav_frame,
                text=text,
                command=command,
                font=("Helvetica", 12),
                bg=Config.ACCENT_COLOR,
                fg="white",
                relief=tk.FLAT,
                padx=20,
                pady=10,
                width=15,
                cursor="hand2"
            )
            btn.pack(fill=tk.X, pady=5)
    
    def create_content_area(self):
        """Create the main content area."""
        self.content_frame = tk.Frame(self.main_container, bg="white")
        self.content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Initial dashboard
        self.show_dashboard()
    
    def create_status_bar(self):
        """Create the status bar."""
        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            font=("Helvetica", 10),
            bg=Config.THEME_COLOR,
            fg="white",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def show_dashboard(self):
        """Show the dashboard view."""
        self.clear_content()
        
        dashboard_frame = tk.Frame(self.content_frame, bg="white")
        dashboard_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Dashboard title
        title = tk.Label(
            dashboard_frame,
            text="System Dashboard",
            font=("Helvetica", 20, "bold"),
            bg="white",
            fg=Config.THEME_COLOR
        )
        title.pack(pady=(0, 20))
        
        # System status
        status_frame = tk.LabelFrame(
            dashboard_frame,
            text="System Status",
            font=("Helvetica", 12, "bold"),
            bg="white",
            fg=Config.THEME_COLOR,
            padx=20,
            pady=20
        )
        status_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Check system status
        status = self.attendance_system.get_system_status()
        is_ready = status.get('initialized', False)
        
        status_text = "System is ready" if is_ready else "System initialization failed"
        status_color = "green" if is_ready else "red"
        
        status_label = tk.Label(
            status_frame,
            text=status_text,
            font=("Helvetica", 12),
            fg=status_color,
            bg="white"
        )
        status_label.pack(anchor=tk.W)
        
        # Statistics
        stats_frame = tk.LabelFrame(
            dashboard_frame,
            text="Statistics",
            font=("Helvetica", 12, "bold"),
            bg="white",
            fg=Config.THEME_COLOR,
            padx=20,
            pady=20
        )
        stats_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Get statistics
        try:
            stats = self.attendance_system.get_statistics()
            training_stats = stats.get('training_stats', {})
            
            stats_text = f"""
            Total Students: {training_stats.get('total_students', 0)}
            Students with Faces: {training_stats.get('students_with_faces', 0)}
            Total Face Encodings: {training_stats.get('total_face_encodings', 0)}
            Coverage: {training_stats.get('coverage_percentage', 0):.1f}%
            """
            
            stats_label = tk.Label(
                stats_frame,
                text=stats_text,
                font=("Helvetica", 11),
                bg="white",
                justify=tk.LEFT
            )
            stats_label.pack(anchor=tk.W)
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            error_label = tk.Label(
                stats_frame,
                text="Error loading statistics",
                font=("Helvetica", 11),
                fg="red",
                bg="white"
            )
            error_label.pack(anchor=tk.W)
        
        # Quick actions
        actions_frame = tk.LabelFrame(
            dashboard_frame,
            text="Quick Actions",
            font=("Helvetica", 12, "bold"),
            bg="white",
            fg=Config.THEME_COLOR,
            padx=20,
            pady=20
        )
        actions_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Action buttons
        actions = [
            ("Register New Student", self.show_registration),
            ("Start Live Attendance", self.show_attendance),
            ("View Reports", self.show_reports)
        ]
        
        for text, command in actions:
            btn = tk.Button(
                actions_frame,
                text=text,
                command=command,
                font=("Helvetica", 10),
                bg=Config.ACCENT_COLOR,
                fg="white",
                relief=tk.FLAT,
                padx=15,
                pady=5
            )
            btn.pack(side=tk.LEFT, padx=(0, 10))
    
    def show_registration(self):
        """Show the student registration view."""
        self.clear_content()
        
        # Import here to avoid circular imports
        from gui.registration_window import RegistrationWindow
        
        registration_frame = RegistrationWindow(self.content_frame, self.attendance_system)
        registration_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    def show_attendance(self):
        """Show the live attendance view."""
        self.clear_content()
        
        # Import here to avoid circular imports
        from gui.attendance_window import AttendanceWindow
        
        attendance_frame = AttendanceWindow(self.content_frame, self.attendance_system)
        attendance_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    def show_reports(self):
        """Show the reports view."""
        self.clear_content()
        
        # Import here to avoid circular imports
        from gui.reports_window import ReportsWindow
        
        reports_frame = ReportsWindow(self.content_frame, self.attendance_system)
        reports_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    def show_settings(self):
        """Show the settings view."""
        self.clear_content()
        
        settings_frame = tk.Frame(self.content_frame, bg="white")
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        title = tk.Label(
            settings_frame,
            text="Settings",
            font=("Helvetica", 20, "bold"),
            bg="white",
            fg=Config.THEME_COLOR
        )
        title.pack(pady=(0, 20))
        
        # Recognition threshold setting
        threshold_frame = tk.LabelFrame(
            settings_frame,
            text="Recognition Settings",
            font=("Helvetica", 12, "bold"),
            bg="white",
            fg=Config.THEME_COLOR,
            padx=20,
            pady=20
        )
        threshold_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(
            threshold_frame,
            text="Recognition Threshold:",
            font=("Helvetica", 11),
            bg="white"
        ).pack(anchor=tk.W)
        
        # Threshold slider
        self.threshold_var = tk.DoubleVar(value=Config.RECOGNITION_THRESHOLD)
        threshold_slider = tk.Scale(
            threshold_frame,
            from_=0.3,
            to=0.8,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self.threshold_var,
            command=self.update_threshold
        )
        threshold_slider.pack(fill=tk.X, pady=(5, 0))
        
        threshold_value = tk.Label(
            threshold_frame,
            text=f"Current: {Config.RECOGNITION_THRESHOLD}",
            font=("Helvetica", 10),
            bg="white"
        )
        threshold_value.pack(anchor=tk.W)
    
    def clear_content(self):
        """Clear the current content area."""
        if self.current_frame:
            self.current_frame.destroy()
        self.current_frame = None
    
    def initialize_components(self):
        """Initialize face recognition components."""
        try:
            # Initialize face detector
            self.detector = create_face_detector()
            
            # Initialize recognition engine
            self.recognition_engine = create_face_recognition_engine()
            
            logger.info("Face recognition components initialized")
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            messagebox.showerror("Initialization Error", f"Failed to initialize face recognition components: {str(e)}")
    
    def update_threshold(self, value):
        """Update the recognition threshold."""
        try:
            new_threshold = float(value)
            Config.update_threshold(new_threshold)
            
            # Update the engine if it exists
            if self.recognition_engine:
                self.recognition_engine.face_matcher.set_tolerance(new_threshold)
            
            logger.info(f"Recognition threshold updated to: {new_threshold}")
            
        except Exception as e:
            logger.error(f"Error updating threshold: {str(e)}")
    
    def update_status(self, message: str, color: str = "black"):
        """Update the status bar."""
        self.status_bar.config(text=message, fg=color)
        self.root.update_idletasks()
    
    def show_error(self, message: str):
        """Show an error message."""
        messagebox.showerror("Error", message)
        logger.error(f"Error displayed: {message}")
    
    def show_info(self, message: str):
        """Show an information message."""
        messagebox.showinfo("Information", message)
        logger.info(f"Info displayed: {message}")
    
    def on_closing(self):
        """Handle window closing."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            # Cleanup
            if self.attendance_system:
                self.attendance_system.cleanup()
            self.root.destroy()


def main():
    """Main entry point for GUI application."""
    try:
        # Create main application
        from main import AttendanceSystem
        app = AttendanceSystem()
        
        # Check if system is properly initialized
        status = app.get_system_status()
        if not status.get('initialized', False):
            print("System failed to initialize. Cannot start GUI.")
            return
        
        # Create and run GUI
        root = tk.Tk()
        gui_app = MainWindow(root, app)
        
        # Handle window closing
        root.protocol("WM_DELETE_WINDOW", gui_app.on_closing)
        
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Error running GUI application: {str(e)}")
        print(f"Failed to start GUI application: {str(e)}")
        print("Please check the logs for more details.")


if __name__ == "__main__":
    main()