"""
GUI module for the Face Recognition Attendance System.

This module provides a Tkinter-based graphical user interface for
student registration, attendance taking, and log management.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import threading
import time
import logging
from PIL import Image, ImageTk
from datetime import datetime
from typing import List, Tuple, Optional

# Configure logging
logger = logging.getLogger(__name__)


class AttendanceGUI:
    """Main GUI application for the attendance system."""
    
    def __init__(self, root: tk.Tk):
        """
        Initialize the GUI application.
        
        Args:
            root (tk.Tk): Root Tkinter window
        """
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # System components (will be injected by main.py)
        self.system = None
        
        # GUI state
        self.is_camera_running = False
        self.camera_thread = None
        self.camera = None
        
        # Initialize GUI components
        self.setup_ui()
        
        # Start status update timer
        self.root.after(1000, self.update_status)
    
    def setup_ui(self):
        """Setup the main user interface."""
        # Main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header_frame = tk.Frame(main_frame, bg='#f0f0f0')
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = tk.Label(
            header_frame, 
            text="Face Recognition Attendance System", 
            font=("Arial", 18, "bold"),
            bg='#f0f0f0',
            fg='#333333'
        )
        title_label.pack(side=tk.LEFT)
        
        # Status indicator
        self.status_label = tk.Label(
            header_frame,
            text="System Status: Initializing...",
            font=("Arial", 10),
            bg='#f0f0f0',
            fg='#666666'
        )
        self.status_label.pack(side=tk.RIGHT)
        
        # Left panel (Controls)
        left_frame = tk.Frame(main_frame, bg='#ffffff', relief=tk.RAISED, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Controls section
        controls_frame = tk.LabelFrame(left_frame, text="Controls", font=("Arial", 12, "bold"), bg='#ffffff')
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Register student button
        self.register_btn = tk.Button(
            controls_frame,
            text="Register Student",
            command=self.register_student,
            font=("Arial", 10, "bold"),
            bg='#4CAF50',
            fg='white',
            width=20,
            height=2
        )
        self.register_btn.pack(pady=10, padx=10, fill=tk.X)
        
        # Start attendance button
        self.attendance_btn = tk.Button(
            controls_frame,
            text="Start Attendance",
            command=self.toggle_attendance,
            font=("Arial", 10, "bold"),
            bg='#2196F3',
            fg='white',
            width=20,
            height=2
        )
        self.attendance_btn.pack(pady=10, padx=10, fill=tk.X)
        
        # View logs button
        self.logs_btn = tk.Button(
            controls_frame,
            text="View Logs",
            command=self.view_logs,
            font=("Arial", 10, "bold"),
            bg='#FF9800',
            fg='white',
            width=20,
            height=2
        )
        self.logs_btn.pack(pady=10, padx=10, fill=tk.X)
        
        # Export logs button
        self.export_btn = tk.Button(
            controls_frame,
            text="Export Logs",
            command=self.export_logs,
            font=("Arial", 10, "bold"),
            bg='#9C27B0',
            fg='white',
            width=20,
            height=2
        )
        self.export_btn.pack(pady=10, padx=10, fill=tk.X)
        
        # Quit button
        self.quit_btn = tk.Button(
            controls_frame,
            text="Quit",
            command=self.quit_application,
            font=("Arial", 10, "bold"),
            bg='#F44336',
            fg='white',
            width=20,
            height=2
        )
        self.quit_btn.pack(pady=10, padx=10, fill=tk.X)
        
        # Statistics section
        stats_frame = tk.LabelFrame(left_frame, text="Statistics", font=("Arial", 12, "bold"), bg='#ffffff')
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Statistics labels
        self.stats_labels = {}
        stats_data = [
            ("Total Students", "0"),
            ("Known Students", "0"),
            ("Today's Attendance", "0"),
            ("Session", "default")
        ]
        
        for label_text, default_value in stats_data:
            frame = tk.Frame(stats_frame, bg='#ffffff')
            frame.pack(fill=tk.X, padx=5, pady=2)
            
            label = tk.Label(frame, text=label_text + ":", font=("Arial", 10), bg='#ffffff', fg='#333333')
            label.pack(side=tk.LEFT)
            
            value_label = tk.Label(frame, text=default_value, font=("Arial", 10, "bold"), bg='#ffffff', fg='#007ACC')
            value_label.pack(side=tk.RIGHT)
            
            self.stats_labels[label_text] = value_label
        
        # Right panel (Video feed)
        right_frame = tk.Frame(main_frame, bg='#ffffff', relief=tk.RAISED, bd=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Video feed section
        video_frame = tk.LabelFrame(right_frame, text="Live Video Feed", font=("Arial", 12, "bold"), bg='#ffffff')
        video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Video canvas
        self.video_canvas = tk.Canvas(video_frame, bg='black', width=640, height=480)
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Instructions
        instructions = tk.Label(
            right_frame,
            text="Instructions: Click 'Start Attendance' to begin live face recognition.\nPress 'q' in the video window to stop.",
            font=("Arial", 9),
            bg='#ffffff',
            fg='#666666',
            justify=tk.LEFT
        )
        instructions.pack(pady=(0, 10))
    
    def set_system(self, system):
        """Set the system instance for GUI operations."""
        self.system = system
        self.update_status()
    
    def update_status(self):
        """Update system status and statistics."""
        try:
            if self.system and self.system.is_initialized:
                status = self.system.get_system_status()
                
                # Update status label
                if status.get('initialized', False):
                    self.status_label.config(text="System Status: ✓ Ready", fg='#4CAF50')
                else:
                    self.status_label.config(text="System Status: ✗ Not Ready", fg='#F44336')
                
                # Update statistics
                stats = status.get('database_stats', {})
                self.stats_labels["Total Students"].config(text=str(stats.get('total_students', 0)))
                self.stats_labels["Known Students"].config(text=str(len(self.system.known_students)))
                
                # Update today's attendance
                today_logs = self.system.db_manager.get_attendance_logs(
                    date=datetime.now().strftime("%Y-%m-%d")
                )
                self.stats_labels["Today's Attendance"].config(text=str(len(today_logs)))
                
                # Update session
                self.stats_labels["Session"].config(text=status.get('default_session', 'default'))
            
            else:
                self.status_label.config(text="System Status: Initializing...", fg='#666666')
        
        except Exception as e:
            logger.error(f"Error updating status: {e}")
            self.status_label.config(text="System Status: Error", fg='#F44336')
        
        # Schedule next update
        self.root.after(5000, self.update_status)
    
    def register_student(self):
        """Handle student registration."""
        if not self.system or not self.system.is_initialized:
            messagebox.showerror("Error", "System not initialized. Please wait for initialization.")
            return
        
        # Get student name
        name_dialog = tk.Toplevel(self.root)
        name_dialog.title("Register Student")
        name_dialog.geometry("300x150")
        name_dialog.configure(bg='#f0f0f0')
        name_dialog.transient(self.root)
        name_dialog.grab_set()
        
        tk.Label(name_dialog, text="Enter student name:", font=("Arial", 10), bg='#f0f0f0').pack(pady=10)
        
        name_entry = tk.Entry(name_dialog, font=("Arial", 10), width=30)
        name_entry.pack(pady=5)
        name_entry.focus()
        
        def on_register():
            name = name_entry.get().strip()
            if not name:
                messagebox.showwarning("Warning", "Please enter a student name.")
                return
            
            name_dialog.destroy()
            
            # Show progress dialog
            progress_dialog = tk.Toplevel(self.root)
            progress_dialog.title("Registration Progress")
            progress_dialog.geometry("300x100")
            progress_dialog.configure(bg='#f0f0f0')
            progress_dialog.transient(self.root)
            progress_dialog.grab_set()
            
            tk.Label(progress_dialog, text=f"Registering {name}...", font=("Arial", 10), bg='#f0f0f0').pack(pady=10)
            progress_bar = ttk.Progressbar(progress_dialog, mode='indeterminate')
            progress_bar.pack(pady=5, padx=20, fill=tk.X)
            progress_bar.start()
            
            def register_thread():
                try:
                    # This would normally capture images and register the student
                    # For now, we'll simulate the process
                    time.sleep(2)  # Simulate image capture
                    
                    # Simulate successful registration
                    result = self.system.register_student(name, 5)
                    
                    progress_dialog.after(0, progress_dialog.destroy)
                    
                    if result['success']:
                        messagebox.showinfo("Success", f"Successfully registered {name}")
                        self.update_status()
                    else:
                        messagebox.showerror("Error", f"Registration failed: {result['error']}")
                
                except Exception as e:
                    progress_dialog.after(0, progress_dialog.destroy)
                    messagebox.showerror("Error", f"Registration error: {str(e)}")
            
            # Run registration in background thread
            threading.Thread(target=register_thread, daemon=True).start()
        
        tk.Button(
            name_dialog, 
            text="Register", 
            command=on_register,
            font=("Arial", 10, "bold"),
            bg='#4CAF50',
            fg='white'
        ).pack(pady=10)
        
        def on_cancel():
            name_dialog.destroy()
        
        tk.Button(
            name_dialog, 
            text="Cancel", 
            command=on_cancel,
            font=("Arial", 10),
            bg='#f0f0f0'
        ).pack(pady=(0, 10))
    
    def toggle_attendance(self):
        """Toggle attendance taking mode."""
        if not self.system or not self.system.is_initialized:
            messagebox.showerror("Error", "System not initialized. Please wait for initialization.")
            return
        
        if self.is_camera_running:
            self.stop_attendance()
        else:
            self.start_attendance()
    
    def start_attendance(self):
        """Start live attendance taking."""
        if not self.system.known_students:
            messagebox.showwarning("Warning", "No known students found. Please register students first.")
            return
        
        self.is_camera_running = True
        self.attendance_btn.config(text="Stop Attendance", bg='#F44336')
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=self.attendance_loop, daemon=True)
        self.camera_thread.start()
    
    def stop_attendance(self):
        """Stop live attendance taking."""
        self.is_camera_running = False
        self.attendance_btn.config(text="Start Attendance", bg='#2196F3')
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        # Clear video canvas
        self.video_canvas.delete("all")
    
    def attendance_loop(self):
        """Main attendance taking loop."""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                messagebox.showerror("Error", "Cannot access camera")
                self.root.after(0, self.stop_attendance)
                return
            
            frame_count = 0
            last_update = time.time()
            
            while self.is_camera_running:
                ret, frame = self.camera.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break
                
                frame_count += 1
                
                # Process frame for recognition
                processed_frame = self.system.face_system.recognize_and_log(
                    frame,
                    self.system.known_students,
                    self.system.db_manager,
                    session="default",
                    tolerance=self.system.face_system.tolerance
                )
                
                # Update GUI with processed frame
                self.root.after(0, self.update_video_display, processed_frame)
                
                # Update statistics every 30 frames
                if frame_count % 30 == 0:
                    current_time = time.time()
                    fps = 30 / (current_time - last_update)
                    last_update = current_time
                    
                    # Update today's attendance count
                    today_logs = self.system.db_manager.get_attendance_logs(
                        date=datetime.now().strftime("%Y-%m-%d")
                    )
                    
                    self.root.after(0, self.update_attendance_count, len(today_logs))
                
                # Check if user pressed 'q' to stop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            self.root.after(0, self.stop_attendance)
        
        except Exception as e:
            logger.error(f"Error in attendance loop: {e}")
            self.root.after(0, self.stop_attendance)
            messagebox.showerror("Error", f"Attendance error: {str(e)}")
    
    def update_video_display(self, frame):
        """Update video display with processed frame."""
        try:
            # Convert OpenCV BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)
            
            # Resize if necessary
            if pil_image.size != (640, 480):
                pil_image = pil_image.resize((640, 480), Image.Resampling.LANCZOS)
            
            # Convert to Tkinter PhotoImage
            photo = ImageTk.PhotoImage(image=pil_image)
            
            # Update canvas
            self.video_canvas.delete("all")
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.video_canvas.image = photo  # Keep reference to prevent garbage collection
        
        except Exception as e:
            logger.error(f"Error updating video display: {e}")
    
    def update_attendance_count(self, count):
        """Update attendance count in statistics."""
        self.stats_labels["Today's Attendance"].config(text=str(count))
    
    def view_logs(self):
        """View attendance logs."""
        if not self.system or not self.system.is_initialized:
            messagebox.showerror("Error", "System not initialized.")
            return
        
        # Create logs window
        logs_window = tk.Toplevel(self.root)
        logs_window.title("Attendance Logs")
        logs_window.geometry("800x500")
        logs_window.configure(bg='#f0f0f0')
        logs_window.transient(self.root)
        
        # Date filter
        filter_frame = tk.Frame(logs_window, bg='#f0f0f0')
        filter_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(filter_frame, text="Date filter (YYYY-MM-DD):", font=("Arial", 10), bg='#f0f0f0').pack(side=tk.LEFT)
        
        date_entry = tk.Entry(filter_frame, font=("Arial", 10), width=15)
        date_entry.pack(side=tk.LEFT, padx=(5, 10))
        
        def load_logs():
            date = date_entry.get().strip()
            if date and not self.validate_date_format(date):
                messagebox.showwarning("Warning", "Invalid date format. Use YYYY-MM-DD.")
                return
            
            logs = self.system.get_attendance_logs(date=date if date else None)
            
            # Clear existing treeview
            for item in tree.get_children():
                tree.delete(item)
            
            # Insert logs
            for log in logs:
                tree.insert("", tk.END, values=log)
            
            # Update count
            count_label.config(text=f"Total Records: {len(logs)}")
        
        tk.Button(
            filter_frame,
            text="Load Logs",
            command=load_logs,
            font=("Arial", 10, "bold"),
            bg='#2196F3',
            fg='white'
        ).pack(side=tk.LEFT)
        
        # Logs display
        tree_frame = tk.Frame(logs_window, bg='#ffffff')
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Treeview for logs
        columns = ("Student ID", "Name", "Timestamp", "Session", "Confidence")
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings")
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True)
        
        # Count label
        count_label = tk.Label(logs_window, text="Total Records: 0", font=("Arial", 10), bg='#f0f0f0')
        count_label.pack(pady=(0, 10))
        
        # Load initial logs
        load_logs()
    
    def validate_date_format(self, date_str):
        """Validate date format YYYY-MM-DD."""
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False
    
    def export_logs(self):
        """Export logs to CSV file."""
        if not self.system or not self.system.is_initialized:
            messagebox.showerror("Error", "System not initialized.")
            return
        
        # Get date filter
        date_dialog = tk.Toplevel(self.root)
        date_dialog.title("Export Logs")
        date_dialog.geometry("300x150")
        date_dialog.configure(bg='#f0f0f0')
        date_dialog.transient(self.root)
        date_dialog.grab_set()
        
        tk.Label(date_dialog, text="Date filter (YYYY-MM-DD, optional):", font=("Arial", 10), bg='#f0f0f0').pack(pady=10)
        
        date_entry = tk.Entry(date_dialog, font=("Arial", 10), width=20)
        date_entry.pack(pady=5)
        
        def on_export():
            date = date_entry.get().strip()
            if date and not self.validate_date_format(date):
                messagebox.showwarning("Warning", "Invalid date format. Use YYYY-MM-DD.")
                return
            
            # Ask for filename
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filename:
                try:
                    success = self.system.export_logs_to_csv(filename, date=date if date else None)
                    date_dialog.destroy()
                    
                    if success:
                        messagebox.showinfo("Success", f"Logs exported to {filename}")
                    else:
                        messagebox.showerror("Error", "Failed to export logs")
                
                except Exception as e:
                    date_dialog.destroy()
                    messagebox.showerror("Error", f"Export error: {str(e)}")
        
        tk.Button(
            date_dialog,
            text="Export",
            command=on_export,
            font=("Arial", 10, "bold"),
            bg='#9C27B0',
            fg='white'
        ).pack(pady=10)
        
        def on_cancel():
            date_dialog.destroy()
        
        tk.Button(
            date_dialog,
            text="Cancel",
            command=on_cancel,
            font=("Arial", 10),
            bg='#f0f0f0'
        ).pack(pady=(0, 10))
    
    def quit_application(self):
        """Quit the application."""
        if messagebox.askyesno("Quit", "Are you sure you want to quit?"):
            if self.camera:
                self.camera.release()
            self.root.quit()
            self.root.destroy()


def create_gui(root, system):
    """
    Create and return the GUI application.
    
    Args:
        root (tk.Tk): Root Tkinter window
        system: System instance
        
    Returns:
        AttendanceGUI: GUI application instance
    """
    gui = AttendanceGUI(root)
    gui.set_system(system)
    return gui