"""
Student registration window for the Face Recognition Attendance System.

This module provides the GUI for registering new students with
face training capabilities.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import threading
import time
from PIL import Image, ImageTk
import numpy as np
import os
from typing import List, Optional

from utils.logger import logger
from utils.helpers import save_image_with_timestamp, validate_student_data
from utils.config import Config


class RegistrationWindow(tk.Frame):
    """Student registration window class."""
    
    def __init__(self, parent: tk.Frame, attendance_system):
        """
        Initialize the registration window.
        
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
        self.current_frame = None
        self.captured_images = []
        
        # Setup UI
        self.setup_ui()
        
        logger.info("Registration window initialized")
    
    def setup_ui(self):
        """Setup the registration window UI."""
        # Title
        title_label = tk.Label(
            self,
            text="Student Registration",
            font=("Helvetica", 18, "bold"),
            bg="white",
            fg="#333333"
        )
        title_label.pack(pady=(0, 20))
        
        # Create main content frame
        content_frame = tk.Frame(self, bg="white")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Left panel: Student information
        left_panel = tk.LabelFrame(
            content_frame,
            text="Student Information",
            font=("Helvetica", 12, "bold"),
            bg="white",
            fg="#333333",
            padx=20,
            pady=20,
            width=400
        )
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        left_panel.pack_propagate(False)
        
        # Right panel: Camera and training
        right_panel = tk.LabelFrame(
            content_frame,
            text="Face Training",
            font=("Helvetica", 12, "bold"),
            bg="white",
            fg="#333333",
            padx=20,
            pady=20
        )
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Setup left panel
        self.setup_student_info_panel(left_panel)
        
        # Setup right panel
        self.setup_training_panel(right_panel)
    
    def setup_student_info_panel(self, parent: tk.LabelFrame):
        """Setup the student information panel."""
        # Name field
        tk.Label(
            parent,
            text="Full Name:",
            font=("Helvetica", 11),
            bg="white"
        ).pack(anchor=tk.W, pady=(0, 5))
        
        self.name_entry = tk.Entry(
            parent,
            font=("Helvetica", 11),
            width=30
        )
        self.name_entry.pack(fill=tk.X, pady=(0, 15))
        
        # Enrollment number field
        tk.Label(
            parent,
            text="Enrollment Number:",
            font=("Helvetica", 11),
            bg="white"
        ).pack(anchor=tk.W, pady=(0, 5))
        
        self.enrollment_entry = tk.Entry(
            parent,
            font=("Helvetica", 11),
            width=30
        )
        self.enrollment_entry.pack(fill=tk.X, pady=(0, 15))
        
        # Image upload section
        tk.Label(
            parent,
            text="Upload Training Images:",
            font=("Helvetica", 11, "bold"),
            bg="white"
        ).pack(anchor=tk.W, pady=(0, 10))
        
        # Upload button
        upload_btn = tk.Button(
            parent,
            text="Browse Images",
            command=self.browse_images,
            font=("Helvetica", 10),
            bg="#3498db",
            fg="white",
            relief=tk.FLAT,
            padx=10,
            pady=5
        )
        upload_btn.pack(pady=(0, 10))
        
        # Selected images list
        self.images_listbox = tk.Listbox(
            parent,
            height=6,
            font=("Helvetica", 10),
            selectmode=tk.SINGLE
        )
        self.images_listbox.pack(fill=tk.X, pady=(0, 10))
        
        # Remove image button
        remove_btn = tk.Button(
            parent,
            text="Remove Selected",
            command=self.remove_selected_image,
            font=("Helvetica", 10),
            bg="#e74c3c",
            fg="white",
            relief=tk.FLAT,
            padx=10,
            pady=5
        )
        remove_btn.pack(pady=(0, 15))
        
        # Clear all button
        clear_btn = tk.Button(
            parent,
            text="Clear All",
            command=self.clear_all_images,
            font=("Helvetica", 10),
            bg="#f39c12",
            fg="white",
            relief=tk.FLAT,
            padx=10,
            pady=5
        )
        clear_btn.pack(pady=(0, 15))
        
        # Register button
        register_btn = tk.Button(
            parent,
            text="Register Student",
            command=self.register_student,
            font=("Helvetica", 12, "bold"),
            bg="#2ecc71",
            fg="white",
            relief=tk.FLAT,
            padx=20,
            pady=10
        )
        register_btn.pack(pady=(20, 0))
    
    def setup_training_panel(self, parent: tk.LabelFrame):
        """Setup the face training panel."""
        # Camera frame
        camera_frame = tk.Frame(parent, bg="black", width=400, height=300)
        camera_frame.pack(pady=(0, 20))
        camera_frame.pack_propagate(False)
        
        # Camera display label
        self.camera_label = tk.Label(camera_frame, bg="black")
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Camera controls
        controls_frame = tk.Frame(parent, bg="white")
        controls_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Start camera button
        self.start_camera_btn = tk.Button(
            controls_frame,
            text="Start Camera",
            command=self.toggle_camera,
            font=("Helvetica", 10),
            bg="#3498db",
            fg="white",
            relief=tk.FLAT,
            padx=10,
            pady=5
        )
        self.start_camera_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Capture image button
        self.capture_btn = tk.Button(
            controls_frame,
            text="Capture Image",
            command=self.capture_image,
            font=("Helvetica", 10),
            bg="#9b59b6",
            fg="white",
            relief=tk.FLAT,
            padx=10,
            pady=5,
            state=tk.DISABLED
        )
        self.capture_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Interactive training button
        self.interactive_btn = tk.Button(
            controls_frame,
            text="Interactive Training",
            command=self.start_interactive_training,
            font=("Helvetica", 10),
            bg="#e67e22",
            fg="white",
            relief=tk.FLAT,
            padx=10,
            pady=5
        )
        self.interactive_btn.pack(side=tk.LEFT)
        
        # Captured images frame
        captured_frame = tk.LabelFrame(
            parent,
            text="Captured Images",
            font=("Helvetica", 10, "bold"),
            bg="white",
            fg="#333333",
            padx=10,
            pady=10
        )
        captured_frame.pack(fill=tk.BOTH, expand=True)
        
        # Captured images list
        self.captured_listbox = tk.Listbox(
            captured_frame,
            height=4,
            font=("Helvetica", 10),
            selectmode=tk.SINGLE
        )
        self.captured_listbox.pack(fill=tk.X, pady=(0, 10))
        
        # Remove captured button
        remove_captured_btn = tk.Button(
            captured_frame,
            text="Remove Selected",
            command=self.remove_captured_image,
            font=("Helvetica", 10),
            bg="#e74c3c",
            fg="white",
            relief=tk.FLAT,
            padx=10,
            pady=5
        )
        remove_captured_btn.pack(pady=(0, 5))
    
    def browse_images(self):
        """Browse and select training images."""
        try:
            file_paths = filedialog.askopenfilenames(
                title="Select Training Images",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
            )
            
            if file_paths:
                for file_path in file_paths:
                    if file_path not in self.captured_images:
                        self.captured_images.append(file_path)
                        self.images_listbox.insert(tk.END, os.path.basename(file_path))
                
                logger.info(f"Added {len(file_paths)} images to training list")
        
        except Exception as e:
            logger.error(f"Error browsing images: {str(e)}")
            messagebox.showerror("Error", f"Failed to browse images: {str(e)}")
    
    def remove_selected_image(self):
        """Remove selected image from the list."""
        try:
            selection = self.images_listbox.curselection()
            if selection:
                index = selection[0]
                self.images_listbox.delete(index)
                del self.captured_images[index]
        
        except Exception as e:
            logger.error(f"Error removing image: {str(e)}")
    
    def clear_all_images(self):
        """Clear all selected images."""
        try:
            self.images_listbox.delete(0, tk.END)
            self.captured_images.clear()
        
        except Exception as e:
            logger.error(f"Error clearing images: {str(e)}")
    
    def remove_captured_image(self):
        """Remove selected captured image."""
        try:
            selection = self.captured_listbox.curselection()
            if selection:
                index = selection[0]
                self.captured_listbox.delete(index)
                # Note: In a real implementation, you might want to delete the actual file
        
        except Exception as e:
            logger.error(f"Error removing captured image: {str(e)}")
    
    def toggle_camera(self):
        """Toggle camera on/off."""
        if not self.camera_active:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start the camera feed."""
        try:
            self.camera = cv2.VideoCapture(Config.CAMERA_INDEX)
            if not self.camera.isOpened():
                messagebox.showerror("Error", "Cannot access camera")
                return
            
            self.camera_active = True
            self.start_camera_btn.config(text="Stop Camera", bg="#e74c3c")
            self.capture_btn.config(state=tk.NORMAL)
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.update_camera_feed)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
            logger.info("Camera started")
        
        except Exception as e:
            logger.error(f"Error starting camera: {str(e)}")
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
    
    def stop_camera(self):
        """Stop the camera feed."""
        try:
            self.camera_active = False
            if self.camera:
                self.camera.release()
            self.start_camera_btn.config(text="Start Camera", bg="#3498db")
            self.capture_btn.config(state=tk.DISABLED)
            
            # Clear camera label
            self.camera_label.config(image="")
            
            logger.info("Camera stopped")
        
        except Exception as e:
            logger.error(f"Error stopping camera: {str(e)}")
    
    def update_camera_feed(self):
        """Update the camera feed display."""
        try:
            while self.camera_active:
                ret, frame = self.camera.read()
                if ret:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Resize frame for display
                    display_frame = cv2.resize(rgb_frame, (400, 300))
                    
                    # Convert to PIL Image
                    pil_image = Image.fromarray(display_frame)
                    photo = ImageTk.PhotoImage(image=pil_image)
                    
                    # Update label
                    self.camera_label.config(image=photo)
                    self.camera_label.image = photo
                
                time.sleep(0.03)  # ~30 FPS
        
        except Exception as e:
            logger.error(f"Error updating camera feed: {str(e)}")
    
    def capture_image(self):
        """Capture an image from the camera."""
        try:
            if not self.camera_active or not self.camera:
                messagebox.showwarning("Warning", "Camera is not active")
                return
            
            ret, frame = self.camera.read()
            if ret:
                # Save the image
                image_path = save_image_with_timestamp(frame, "training")
                self.captured_images.append(image_path)
                self.images_listbox.insert(tk.END, os.path.basename(image_path))
                
                # Add to captured listbox
                self.captured_listbox.insert(tk.END, f"Captured: {os.path.basename(image_path)}")
                
                logger.info(f"Image captured: {image_path}")
            else:
                messagebox.showerror("Error", "Failed to capture image")
        
        except Exception as e:
            logger.error(f"Error capturing image: {str(e)}")
            messagebox.showerror("Error", f"Failed to capture image: {str(e)}")
    
    def start_interactive_training(self):
        """Start interactive training with camera."""
        try:
            # This would open a new window for interactive training
            # For now, show a message
            messagebox.showinfo(
                "Interactive Training", 
                "Interactive training would start here.\n\n"
                "Instructions:\n"
                "1. Position yourself in front of the camera\n"
                "2. Press SPACE to capture samples\n"
                "3. Capture 5-10 different angles\n"
                "4. Press ESC to finish"
            )
        
        except Exception as e:
            logger.error(f"Error starting interactive training: {str(e)}")
            messagebox.showerror("Error", f"Failed to start interactive training: {str(e)}")
    
    def register_student(self):
        """Register the student with face training."""
        try:
            # Get student information
            name = self.name_entry.get().strip()
            enrollment_number = self.enrollment_entry.get().strip()
            
            # Validate input
            is_valid, error_msg = validate_student_data(name, enrollment_number)
            if not is_valid:
                messagebox.showerror("Validation Error", error_msg)
                return
            
            # Check if images are provided
            if not self.captured_images:
                if not messagebox.askyesno("No Images", "No training images provided. Continue with camera training only?"):
                    return
            
            # Show progress
            progress_window = tk.Toplevel(self)
            progress_window.title("Registration Progress")
            progress_window.geometry("300x150")
            progress_window.transient(self)
            progress_window.grab_set()
            
            progress_label = tk.Label(
                progress_window,
                text="Registering student...",
                font=("Helvetica", 12),
                bg="white"
            )
            progress_label.pack(pady=20)
            
            progress_bar = ttk.Progressbar(
                progress_window,
                mode='indeterminate',
                length=200
            )
            progress_bar.pack(pady=10)
            progress_bar.start()
            
            # Run registration in background thread
            def register_thread():
                try:
                    # Register student
                    result = self.attendance_system.register_student(
                        name, enrollment_number, self.captured_images, 0
                    )
                    
                    # Update UI on main thread
                    self.root.after(0, lambda: self.handle_registration_result(result, progress_window))
                
                except Exception as e:
                    self.root.after(0, lambda: self.handle_registration_error(str(e), progress_window))
            
            threading.Thread(target=register_thread, daemon=True).start()
        
        except Exception as e:
            logger.error(f"Error in registration: {str(e)}")
            messagebox.showerror("Error", f"Registration failed: {str(e)}")
    
    def handle_registration_result(self, result: dict, progress_window: tk.Toplevel):
        """Handle registration result."""
        try:
            progress_window.destroy()
            
            if result.get('success', False):
                messagebox.showinfo(
                    "Success", 
                    f"Student {result['name']} registered successfully!\n"
                    f"Student ID: {result['student_id']}"
                )
                
                # Clear form
                self.name_entry.delete(0, tk.END)
                self.enrollment_entry.delete(0, tk.END)
                self.clear_all_images()
                self.captured_listbox.delete(0, tk.END)
                
            else:
                error_msg = result.get('error', 'Unknown error')
                messagebox.showerror("Registration Failed", error_msg)
        
        except Exception as e:
            logger.error(f"Error handling registration result: {str(e)}")
    
    def handle_registration_error(self, error_msg: str, progress_window: tk.Toplevel):
        """Handle registration error."""
        try:
            progress_window.destroy()
            messagebox.showerror("Registration Error", error_msg)
        
        except Exception as e:
            logger.error(f"Error handling registration error: {str(e)}")
    
    def destroy(self):
        """Clean up when destroying the window."""
        try:
            self.stop_camera()
            super().destroy()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")