"""
GUI module for the Face Recognition Attendance System.

This module provides a Tkinter-based graphical user interface for
student registration, attendance taking, and log management.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import logging
from datetime import datetime

from config import Config
from recognition import face_backend_available, recognize_and_log

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None  # type: ignore
    ImageTk = None  # type: ignore

# Configure logging
logger = logging.getLogger(__name__)


class AttendanceGUI:
    """Main GUI application for the attendance system."""

    COLORS = {
        "background": "#f5f7fb",
        "panel": "#ffffff",
        "panel_border": "#d9dee8",
        "text": "#1f2937",
        "muted": "#6b7280",
        "primary": "#2563eb",
        "primary_hover": "#1d4ed8",
        "success": "#16a34a",
        "warning": "#d97706",
        "danger": "#dc2626",
        "video": "#0b1020",
        "video_border": "#1f2937",
        "soft_blue": "#e8f0ff",
    }
    
    def __init__(self, root: tk.Tk):
        """
        Initialize the GUI application.
        
        Args:
            root (tk.Tk): Root Tkinter window
        """
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry(f"{Config.GUI_WIDTH}x{Config.GUI_HEIGHT}")
        self.root.minsize(980, 640)
        self.root.configure(bg=self.COLORS["background"])
        
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
        main_frame = tk.Frame(self.root, bg=self.COLORS["background"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=28, pady=24)

        header_frame = tk.Frame(main_frame, bg=self.COLORS["background"])
        header_frame.pack(fill=tk.X, pady=(0, 20))

        title_group = tk.Frame(header_frame, bg=self.COLORS["background"])
        title_group.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(
            title_group,
            text="Attendance Register",
            font=("Arial", 28, "bold"),
            bg=self.COLORS["background"],
            fg=self.COLORS["text"],
            anchor=tk.W,
        ).pack(anchor=tk.W)

        tk.Label(
            title_group,
            text="Local face attendance with lightweight manual fallback",
            font=("Arial", 12),
            bg=self.COLORS["background"],
            fg=self.COLORS["muted"],
            anchor=tk.W,
        ).pack(anchor=tk.W, pady=(4, 0))

        self.status_label = tk.Label(
            header_frame,
            text="Initializing",
            font=("Arial", 11, "bold"),
            bg="#eef2f7",
            fg=self.COLORS["muted"],
            padx=16,
            pady=8,
        )
        self.status_label.pack(side=tk.RIGHT)

        body_frame = tk.Frame(main_frame, bg=self.COLORS["background"])
        body_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = tk.Frame(
            body_frame,
            bg=self.COLORS["panel"],
            highlightthickness=1,
            highlightbackground=self.COLORS["panel_border"],
            width=280,
        )
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 18))
        left_frame.pack_propagate(False)

        tk.Label(
            left_frame,
            text="Actions",
            font=("Arial", 13, "bold"),
            bg=self.COLORS["panel"],
            fg=self.COLORS["text"],
            anchor=tk.W,
        ).pack(fill=tk.X, padx=18, pady=(20, 10))

        self.register_btn = self._make_action_button(left_frame, "Register student", self.register_student)
        self.register_btn.pack(fill=tk.X, padx=18, pady=5)

        self.attendance_btn = self._make_action_button(left_frame, "Start attendance", self.toggle_attendance, primary=True)
        self.attendance_btn.pack(fill=tk.X, padx=18, pady=5)

        self.logs_btn = self._make_action_button(left_frame, "View logs", self.view_logs)
        self.logs_btn.pack(fill=tk.X, padx=18, pady=5)

        self.people_btn = self._make_action_button(left_frame, "Manage people", self.manage_people)
        self.people_btn.pack(fill=tk.X, padx=18, pady=5)

        self.export_btn = self._make_action_button(left_frame, "Export logs", self.export_logs)
        self.export_btn.pack(fill=tk.X, padx=18, pady=5)

        self.quit_btn = self._make_action_button(left_frame, "Quit", self.quit_application, danger=True)
        self.quit_btn.pack(fill=tk.X, padx=18, pady=(5, 18))

        tk.Frame(left_frame, bg=self.COLORS["panel_border"], height=1).pack(fill=tk.X, padx=18, pady=(0, 18))

        tk.Label(
            left_frame,
            text="Today",
            font=("Arial", 13, "bold"),
            bg=self.COLORS["panel"],
            fg=self.COLORS["text"],
            anchor=tk.W,
        ).pack(fill=tk.X, padx=18, pady=(0, 10))

        self.stats_labels = {}
        stats_frame = tk.Frame(left_frame, bg=self.COLORS["panel"])
        stats_frame.pack(fill=tk.X, padx=18)

        for label_text, default_value in [
            ("Total Students", "0"),
            ("Known Students", "0"),
            ("Today's Attendance", "0"),
            ("Session", "default"),
        ]:
            self._make_stat_row(stats_frame, label_text, default_value)

        self.backend_label = tk.Label(
            left_frame,
            text="Checking backend...",
            font=("Arial", 10),
            bg=self.COLORS["soft_blue"],
            fg=self.COLORS["text"],
            justify=tk.LEFT,
            wraplength=220,
            padx=12,
            pady=10,
        )
        self.backend_label.pack(fill=tk.X, padx=18, pady=(22, 0))

        right_frame = tk.Frame(
            body_frame,
            bg=self.COLORS["panel"],
            highlightthickness=1,
            highlightbackground=self.COLORS["panel_border"],
        )
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        video_header = tk.Frame(right_frame, bg=self.COLORS["panel"])
        video_header.pack(fill=tk.X, padx=22, pady=(20, 12))

        tk.Label(
            video_header,
            text="Camera",
            font=("Arial", 16, "bold"),
            bg=self.COLORS["panel"],
            fg=self.COLORS["text"],
        ).pack(side=tk.LEFT)

        self.video_state_label = tk.Label(
            video_header,
            text="Preview idle",
            font=("Arial", 10, "bold"),
            bg="#f3f4f6",
            fg=self.COLORS["muted"],
            padx=12,
            pady=5,
        )
        self.video_state_label.pack(side=tk.RIGHT)

        video_shell = tk.Frame(right_frame, bg=self.COLORS["video_border"])
        video_shell.pack(fill=tk.BOTH, expand=True, padx=22, pady=(0, 16))

        self.video_canvas = tk.Canvas(
            video_shell,
            bg=self.COLORS["video"],
            highlightthickness=0,
            width=760,
            height=480,
        )
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        self.video_canvas.bind("<Configure>", lambda _event: self.draw_video_placeholder())
        self.draw_video_placeholder()

        self.instructions_label = tk.Label(
            right_frame,
            text="Automatic recognition is optional. When it is unavailable, use manual attendance from the CLI.",
            font=("Arial", 10),
            bg=self.COLORS["panel"],
            fg=self.COLORS["muted"],
            anchor=tk.W,
        )
        self.instructions_label.pack(fill=tk.X, padx=22, pady=(0, 18))

    def _make_action_button(self, parent, text, command, primary=False, danger=False):
        bg = self.COLORS["primary"] if primary else self.COLORS["panel"]
        fg = self.COLORS["primary"] if primary else self.COLORS["text"]
        if primary:
            bg = self.COLORS["soft_blue"]
        if danger:
            bg = "#fff1f2"
            fg = self.COLORS["danger"]
        button = tk.Button(
            parent,
            text=text,
            command=command,
            font=("Arial", 11, "bold"),
            bg=bg,
            fg=fg,
            activebackground="#dbeafe" if primary else "#eef2f7",
            activeforeground=self.COLORS["primary"] if primary else self.COLORS["text"],
            relief=tk.FLAT,
            bd=0,
            highlightthickness=1,
            highlightbackground=self.COLORS["panel_border"],
            cursor="hand2",
            anchor=tk.W,
            padx=14,
            pady=11,
        )
        return button

    def _make_stat_row(self, parent, label_text, default_value):
        row = tk.Frame(parent, bg=self.COLORS["panel"])
        row.pack(fill=tk.X, pady=6)

        tk.Label(
            row,
            text=label_text,
            font=("Arial", 10),
            bg=self.COLORS["panel"],
            fg=self.COLORS["muted"],
            anchor=tk.W,
        ).pack(side=tk.LEFT)

        value_label = tk.Label(
            row,
            text=default_value,
            font=("Arial", 12, "bold"),
            bg=self.COLORS["panel"],
            fg=self.COLORS["text"],
            anchor=tk.E,
        )
        value_label.pack(side=tk.RIGHT)
        self.stats_labels[label_text] = value_label

    def draw_video_placeholder(self):
        """Draw the idle state inside the camera canvas."""
        if not hasattr(self, "video_canvas"):
            return
        self.video_canvas.delete("all")
        width = max(1, self.video_canvas.winfo_width())
        height = max(1, self.video_canvas.winfo_height())
        self.video_canvas.create_text(
            width // 2,
            height // 2 - 16,
            text="Camera preview",
            fill="#e5e7eb",
            font=("Arial", 20, "bold"),
        )
        self.video_canvas.create_text(
            width // 2,
            height // 2 + 20,
            text="Start attendance to open the camera",
            fill="#9ca3af",
            font=("Arial", 12),
        )

    def _configure_dialog(self, dialog, title, width=420, height=240):
        dialog.title(title)
        dialog.geometry(f"{width}x{height}")
        dialog.configure(bg=self.COLORS["background"])
        dialog.transient(self.root)
        dialog.grab_set()
        return dialog

    def _dialog_panel(self, dialog, padx=22, pady=20):
        panel = tk.Frame(
            dialog,
            bg=self.COLORS["panel"],
            highlightthickness=1,
            highlightbackground=self.COLORS["panel_border"],
        )
        panel.pack(fill=tk.BOTH, expand=True, padx=padx, pady=pady)
        return panel

    def _make_dialog_button(self, parent, text, command, primary=False, danger=False):
        return self._make_action_button(parent, text, command, primary=primary, danger=danger)
    
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
                    self.status_label.config(text="Ready", bg="#ecfdf5", fg=self.COLORS["success"])
                else:
                    self.status_label.config(text="Not ready", bg="#fef2f2", fg=self.COLORS["danger"])
                
                # Update statistics
                stats = status.get('database_stats', {})
                self.stats_labels["Total Students"].config(text=str(stats.get('total_students', 0)))
                self.stats_labels["Known Students"].config(text=str(len(self.system.known_students)))
                
                # Update today's attendance
                today_logs = self.system.db_manager.get_attendance_logs(
                    date=datetime.now().strftime("%Y-%m-%d")
                )
                self.stats_labels["Today's Attendance"].config(text=str(len(today_logs)))
                
                self.stats_labels["Session"].config(text=status.get('default_session', Config.DEFAULT_SESSION))
                if status.get("full_ready"):
                    self.backend_label.config(
                        text="Automatic recognition is available.",
                        bg="#ecfdf5",
                        fg=self.COLORS["success"],
                    )
                else:
                    self.backend_label.config(
                        text="Automatic recognition unavailable. Manual attendance is available from CLI mode.",
                        bg=self.COLORS["soft_blue"],
                        fg=self.COLORS["text"],
                    )
            
            else:
                self.status_label.config(text="Initializing", bg="#eef2f7", fg=self.COLORS["muted"])
        
        except Exception as e:
            logger.error(f"Error updating status: {e}")
            self.status_label.config(text="Error", bg="#fef2f2", fg=self.COLORS["danger"])
        
        # Schedule next update
        self.root.after(5000, self.update_status)
    
    def register_student(self):
        """Handle student registration."""
        if not self.system or not self.system.is_initialized:
            messagebox.showerror("Error", "System not initialized. Please wait for initialization.")
            return
        
        name_dialog = tk.Toplevel(self.root)
        self._configure_dialog(name_dialog, "Register Student", width=540, height=620)
        panel = self._dialog_panel(name_dialog)
        selected_images = []
        
        tk.Label(
            panel,
            text="Register student",
            font=("Arial", 16, "bold"),
            bg=self.COLORS["panel"],
            fg=self.COLORS["text"],
            anchor=tk.W,
        ).pack(fill=tk.X, padx=18, pady=(18, 6))

        def field(label_text):
            tk.Label(
                panel,
                text=label_text,
                font=("Arial", 10, "bold"),
                bg=self.COLORS["panel"],
                fg=self.COLORS["muted"],
                anchor=tk.W,
            ).pack(fill=tk.X, padx=18, pady=(8, 4))
            entry = tk.Entry(
                panel,
                font=("Arial", 12),
                relief=tk.FLAT,
                highlightthickness=1,
                highlightbackground=self.COLORS["panel_border"],
                highlightcolor=self.COLORS["primary"],
            )
            entry.pack(fill=tk.X, padx=18, pady=(0, 8), ipady=8)
            return entry

        name_entry = field("Name")
        enrollment_entry = field("Staff/student ID (optional)")
        category_entry = field("Class / department (optional)")
        tk.Label(
            panel,
            text="Use this for classes, departments, shifts, or staff groups, e.g. Class 1A.",
            font=("Arial", 9),
            bg=self.COLORS["panel"],
            fg=self.COLORS["muted"],
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=460,
        ).pack(fill=tk.X, padx=18, pady=(0, 8))
        name_entry.focus()

        image_label = tk.Label(
            panel,
            text="No photos selected. Use camera capture, or select clear face photos for automatic recognition training.",
            font=("Arial", 9),
            bg=self.COLORS["panel"],
            fg=self.COLORS["muted"],
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=450,
        )
        image_label.pack(fill=tk.X, padx=18, pady=(6, 8))

        def choose_images():
            paths = filedialog.askopenfilenames(
                title="Select training photos",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                    ("All files", "*.*"),
                ],
            )
            if paths:
                selected_images.clear()
                selected_images.extend(paths)
                image_label.config(text=f"{len(selected_images)} photo(s) selected for this person.")

        def use_camera_capture():
            selected_images.clear()
            image_label.config(text="Camera capture will start after Register if automatic recognition is available.")

        photo_buttons = tk.Frame(panel, bg=self.COLORS["panel"])
        photo_buttons.pack(fill=tk.X, padx=18, pady=(0, 12))
        self._make_dialog_button(photo_buttons, "Select photos", choose_images, primary=True).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self._make_dialog_button(photo_buttons, "Use camera", use_camera_capture).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
        
        def on_register():
            name = name_entry.get().strip()
            if not name:
                messagebox.showwarning("Warning", "Please enter a student name.")
                return
            enrollment_number = enrollment_entry.get().strip() or None
            category = category_entry.get().strip()
            image_paths = list(selected_images) or None
            
            name_dialog.destroy()
            
            # Show progress dialog
            progress_dialog = tk.Toplevel(self.root)
            self._configure_dialog(progress_dialog, "Registration Progress", width=360, height=150)
            progress_panel = self._dialog_panel(progress_dialog)
            
            tk.Label(
                progress_panel,
                text=f"Registering {name}...",
                font=("Arial", 12, "bold"),
                bg=self.COLORS["panel"],
                fg=self.COLORS["text"],
            ).pack(pady=(24, 10))
            progress_bar = ttk.Progressbar(progress_dialog, mode='indeterminate')
            progress_bar.pack(pady=5, padx=42, fill=tk.X)
            progress_bar.start()
            
            def register_thread():
                try:
                    result = self.system.register_student(
                        name,
                        enrollment_number=enrollment_number,
                        category=category,
                        image_paths=image_paths,
                        num_images=5,
                        allow_manual_without_face=True,
                    )

                    def finish_registration():
                        progress_dialog.destroy()
                        if result['success']:
                            details = f"Successfully registered {name}"
                            if result.get("manual_only"):
                                details += "\n\nAutomatic recognition data was not created, but the record is available for manual attendance."
                            messagebox.showinfo("Success", details)
                            self.update_status()
                        else:
                            messagebox.showerror("Error", f"Registration failed: {result['error']}")

                    self.root.after(0, finish_registration)
                
                except Exception as e:
                    error_message = str(e)

                    def fail_registration():
                        progress_dialog.destroy()
                        messagebox.showerror("Error", f"Registration error: {error_message}")

                    self.root.after(0, fail_registration)
            
            # Run registration in background thread
            threading.Thread(target=register_thread, daemon=True).start()
        
        def on_cancel():
            name_dialog.destroy()

        buttons = tk.Frame(panel, bg=self.COLORS["panel"])
        buttons.pack(fill=tk.X, padx=18, pady=(0, 18))
        self._make_dialog_button(buttons, "Register", on_register, primary=True).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self._make_dialog_button(buttons, "Cancel", on_cancel).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
    
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
        if cv2 is None:
            messagebox.showerror("Missing Dependency", "OpenCV is not installed. Install requirements-minimal.txt.")
            return
        
        self.is_camera_running = True
        self.attendance_btn.config(
            text="Stop attendance",
            bg="#fff1f2",
            fg=self.COLORS["danger"],
            activebackground="#ffe4e6",
            activeforeground=self.COLORS["danger"],
        )
        self.video_state_label.config(text="Camera running", bg="#ecfdf5", fg=self.COLORS["success"])
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=self.attendance_loop, daemon=True)
        self.camera_thread.start()
    
    def stop_attendance(self):
        """Stop live attendance taking."""
        self.is_camera_running = False
        self.attendance_btn.config(
            text="Start attendance",
            bg=self.COLORS["soft_blue"],
            fg=self.COLORS["primary"],
            activebackground="#dbeafe",
            activeforeground=self.COLORS["primary"],
        )
        self.video_state_label.config(text="Preview idle", bg="#f3f4f6", fg=self.COLORS["muted"])
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        self.draw_video_placeholder()
    
    def attendance_loop(self):
        """Main attendance taking loop."""
        try:
            self.camera = cv2.VideoCapture(Config.CAMERA_INDEX)
            if not self.camera.isOpened():
                self.root.after(0, lambda: messagebox.showerror("Error", f"Cannot access camera {Config.CAMERA_INDEX}. Check permissions or ATTENDANCE_CAMERA_INDEX."))
                self.root.after(0, self.stop_attendance)
                return
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
            self.camera.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
            
            frame_count = 0
            last_update = time.time()
            last_processed_frame = None
            
            while self.is_camera_running:
                ret, frame = self.camera.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break
                
                frame_count += 1
                
                if frame_count % Config.RECOGNITION_INTERVAL == 0:
                    processed_frame = recognize_and_log(
                        frame,
                        self.system.known_students,
                        self.system.db_manager,
                        session=Config.DEFAULT_SESSION,
                        tolerance=self.system.face_system.tolerance,
                        known_encodings=self.system.face_system.known_encodings,
                        known_names=self.system.face_system.known_names,
                        known_ids=self.system.face_system.known_ids,
                    )
                    last_processed_frame = processed_frame
                else:
                    processed_frame = last_processed_frame if last_processed_frame is not None else frame
                
                # Update GUI with processed frame
                self.root.after(0, self.update_video_display, processed_frame)
                
                # Update statistics every 30 frames
                if frame_count % 30 == 0:
                    current_time = time.time()
                    fps = 30 / max(0.001, (current_time - last_update))
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
            error_message = str(e)
            logger.error(f"Error in attendance loop: {error_message}")
            self.root.after(0, self.stop_attendance)
            self.root.after(0, lambda: messagebox.showerror("Error", f"Attendance error: {error_message}"))
    
    def update_video_display(self, frame):
        """Update video display with processed frame."""
        try:
            if cv2 is None or Image is None or ImageTk is None:
                return
            # Convert OpenCV BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)
            
            # Resize if necessary
            canvas_width = max(1, self.video_canvas.winfo_width())
            canvas_height = max(1, self.video_canvas.winfo_height())
            image_width, image_height = pil_image.size
            scale = max(canvas_width / image_width, canvas_height / image_height)
            new_size = (max(1, int(image_width * scale)), max(1, int(image_height * scale)))
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            left = max(0, (new_size[0] - canvas_width) // 2)
            top = max(0, (new_size[1] - canvas_height) // 2)
            pil_image = pil_image.crop((left, top, left + canvas_width, top + canvas_height))
            
            # Convert to Tkinter PhotoImage
            photo = ImageTk.PhotoImage(image=pil_image)
            
            # Update canvas
            self.video_canvas.delete("all")
            self.video_canvas.create_image(
                0,
                0,
                anchor=tk.NW,
                image=photo,
            )
            self.video_canvas.image = photo  # Keep reference to prevent garbage collection
        
        except Exception as e:
            logger.error(f"Error updating video display: {e}")
    
    def update_attendance_count(self, count):
        """Update attendance count in statistics."""
        self.stats_labels["Today's Attendance"].config(text=str(count))

    def manage_people(self):
        """View and delete registered staff/students."""
        if not self.system or not self.system.is_initialized:
            messagebox.showerror("Error", "System not initialized.")
            return

        people_window = tk.Toplevel(self.root)
        people_window.title("Registered People")
        people_window.geometry("980x640")
        people_window.configure(bg=self.COLORS["background"])
        people_window.transient(self.root)

        header = tk.Frame(people_window, bg=self.COLORS["background"])
        header.pack(fill=tk.X, padx=24, pady=(22, 14))

        tk.Label(
            header,
            text="Registered People",
            font=("Arial", 20, "bold"),
            bg=self.COLORS["background"],
            fg=self.COLORS["text"],
        ).pack(side=tk.LEFT)

        actions = tk.Frame(header, bg=self.COLORS["background"])
        actions.pack(side=tk.RIGHT)

        table_frame = tk.Frame(
            people_window,
            bg=self.COLORS["panel"],
            highlightthickness=1,
            highlightbackground=self.COLORS["panel_border"],
        )
        table_frame.pack(fill=tk.BOTH, expand=True, padx=24, pady=(0, 14))

        columns = ("ID", "Name", "Staff/Student ID", "Class/Department", "Photos", "Face Samples")
        style = ttk.Style(people_window)
        style.configure("People.Treeview", rowheight=30, font=("Arial", 11))
        style.configure("People.Treeview.Heading", font=("Arial", 11, "bold"))
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")
        tree.configure(style="People.Treeview")

        widths = {
            "ID": 70,
            "Name": 220,
            "Staff/Student ID": 170,
            "Class/Department": 190,
            "Photos": 90,
            "Face Samples": 120,
        }
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=widths[col], anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True)

        count_label = tk.Label(
            people_window,
            text="Total people: 0",
            font=("Arial", 11, "bold"),
            bg=self.COLORS["background"],
            fg=self.COLORS["muted"],
        )
        count_label.pack(pady=(0, 18))

        def load_people():
            for item in tree.get_children():
                tree.delete(item)
            people = self.system.list_registered_people()
            for person in people:
                tree.insert(
                    "",
                    tk.END,
                    iid=str(person["id"]),
                    values=(
                        person["id"],
                        person["name"],
                        person["enrollment_number"],
                        person["category"] or "-",
                        person["image_count"],
                        person["embedding_count"],
                    ),
                )
            count_label.config(text=f"Total people: {len(people)}")

        def delete_selected():
            selected = tree.selection()
            if not selected:
                messagebox.showwarning("No selection", "Select a person to delete.")
                return
            person_id = int(selected[0])
            values = tree.item(selected[0], "values")
            name = values[1] if values else "this person"
            if not messagebox.askyesno("Delete registration", f"Delete {name} and their attendance records?"):
                return
            if self.system.delete_registered_person(person_id):
                load_people()
                self.update_status()
            else:
                messagebox.showerror("Delete failed", "Could not delete the selected registration.")

        self._make_dialog_button(actions, "Refresh", load_people).pack(side=tk.LEFT, padx=(0, 8))
        self._make_dialog_button(actions, "Delete selected", delete_selected, danger=True).pack(side=tk.LEFT)
        load_people()
    
    def view_logs(self):
        """View attendance logs."""
        if not self.system or not self.system.is_initialized:
            messagebox.showerror("Error", "System not initialized.")
            return
        
        # Create logs window
        logs_window = tk.Toplevel(self.root)
        logs_window.title("Attendance Logs")
        logs_window.geometry("980x640")
        logs_window.configure(bg=self.COLORS["background"])
        logs_window.transient(self.root)
        
        # Date filter
        header = tk.Frame(logs_window, bg=self.COLORS["background"])
        header.pack(fill=tk.X, padx=24, pady=(22, 14))
        
        tk.Label(
            header,
            text="Attendance Logs",
            font=("Arial", 20, "bold"),
            bg=self.COLORS["background"],
            fg=self.COLORS["text"],
        ).pack(side=tk.LEFT)

        filter_frame = tk.Frame(header, bg=self.COLORS["background"])
        filter_frame.pack(side=tk.RIGHT)

        tk.Label(
            filter_frame,
            text="Date",
            font=("Arial", 10, "bold"),
            bg=self.COLORS["background"],
            fg=self.COLORS["muted"],
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        date_entry = tk.Entry(
            filter_frame,
            font=("Arial", 11),
            width=14,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=self.COLORS["panel_border"],
            highlightcolor=self.COLORS["primary"],
        )
        date_entry.pack(side=tk.LEFT, padx=(0, 10), ipady=6)
        
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
            count_label.config(text=f"Total records: {len(logs)}")
        
        self._make_dialog_button(filter_frame, "Load", load_logs, primary=True).pack(side=tk.LEFT)
        
        # Logs display
        tree_frame = tk.Frame(
            logs_window,
            bg=self.COLORS["panel"],
            highlightthickness=1,
            highlightbackground=self.COLORS["panel_border"],
        )
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=24, pady=(0, 14))
        
        # Treeview for logs
        columns = ("Student ID", "Name", "Timestamp", "Session", "Confidence")
        style = ttk.Style(logs_window)
        style.configure("Attendance.Treeview", rowheight=30, font=("Arial", 11))
        style.configure("Attendance.Treeview.Heading", font=("Arial", 11, "bold"))
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings")
        tree.configure(style="Attendance.Treeview")
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150, anchor=tk.CENTER)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True)
        
        # Count label
        count_label = tk.Label(
            logs_window,
            text="Total records: 0",
            font=("Arial", 11, "bold"),
            bg=self.COLORS["background"],
            fg=self.COLORS["muted"],
        )
        count_label.pack(pady=(0, 18))
        
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
        self._configure_dialog(date_dialog, "Export Logs", width=430, height=260)
        panel = self._dialog_panel(date_dialog)
        
        tk.Label(
            panel,
            text="Export logs",
            font=("Arial", 16, "bold"),
            bg=self.COLORS["panel"],
            fg=self.COLORS["text"],
            anchor=tk.W,
        ).pack(fill=tk.X, padx=18, pady=(18, 6))

        tk.Label(
            panel,
            text="Date filter (optional)",
            font=("Arial", 10, "bold"),
            bg=self.COLORS["panel"],
            fg=self.COLORS["muted"],
            anchor=tk.W,
        ).pack(fill=tk.X, padx=18, pady=(8, 4))
        
        date_entry = tk.Entry(
            panel,
            font=("Arial", 12),
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=self.COLORS["panel_border"],
            highlightcolor=self.COLORS["primary"],
        )
        date_entry.pack(fill=tk.X, padx=18, pady=(0, 6), ipady=8)

        tk.Label(
            panel,
            text="Use YYYY-MM-DD or leave blank for all records.",
            font=("Arial", 9),
            bg=self.COLORS["panel"],
            fg=self.COLORS["muted"],
            anchor=tk.W,
        ).pack(fill=tk.X, padx=18, pady=(0, 16))
        
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
        
        def on_cancel():
            date_dialog.destroy()

        buttons = tk.Frame(panel, bg=self.COLORS["panel"])
        buttons.pack(fill=tk.X, padx=18, pady=(0, 18))
        self._make_dialog_button(buttons, "Export", on_export, primary=True).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self._make_dialog_button(buttons, "Cancel", on_cancel).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
    
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
