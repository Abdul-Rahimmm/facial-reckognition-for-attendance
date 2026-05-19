"""
Reports window for the Face Recognition Attendance System.

This module provides the GUI for viewing and exporting attendance reports.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
from datetime import datetime, timedelta
import threading
from typing import List, Dict, Optional

from utils.logger import logger
from utils.config import Config


class ReportsWindow(tk.Frame):
    """Reports window class."""
    
    def __init__(self, parent: tk.Frame, attendance_system):
        """
        Initialize the reports window.
        
        Args:
            parent (tk.Frame): Parent frame
            attendance_system: Main attendance system instance
        """
        super().__init__(parent, bg="white")
        self.attendance_system = attendance_system
        
        # Initialize variables
        self.current_report_data = []
        
        # Setup UI
        self.setup_ui()
        
        # Load initial data
        self.load_recent_attendance()
        
        logger.info("Reports window initialized")
    
    def setup_ui(self):
        """Setup the reports window UI."""
        # Title
        title_label = tk.Label(
            self,
            text="Attendance Reports",
            font=("Helvetica", 18, "bold"),
            bg="white",
            fg="#333333"
        )
        title_label.pack(pady=(0, 20))
        
        # Create main content frame
        content_frame = tk.Frame(self, bg="white")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Top panel: Filters and controls
        top_panel = tk.LabelFrame(
            content_frame,
            text="Report Filters",
            font=("Helvetica", 12, "bold"),
            bg="white",
            fg="#333333",
            padx=20,
            pady=20
        )
        top_panel.pack(fill=tk.X, pady=(0, 20))
        
        # Bottom panel: Data display and export
        bottom_panel = tk.Frame(content_frame, bg="white")
        bottom_panel.pack(fill=tk.BOTH, expand=True)
        
        # Setup top panel
        self.setup_filters_panel(top_panel)
        
        # Setup bottom panel
        self.setup_data_panel(bottom_panel)
        self.setup_export_panel(bottom_panel)
    
    def setup_filters_panel(self, parent: tk.LabelFrame):
        """Setup the filters panel."""
        # Date range frame
        date_frame = tk.Frame(parent, bg="white")
        date_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Start date
        tk.Label(date_frame, text="Start Date:", font=("Helvetica", 11), bg="white").pack(side=tk.LEFT, padx=(0, 10))
        
        self.start_date_var = tk.StringVar(value=(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"))
        self.start_date_entry = tk.Entry(date_frame, textvariable=self.start_date_var, font=("Helvetica", 11), width=12)
        self.start_date_entry.pack(side=tk.LEFT, padx=(0, 20))
        
        # End date
        tk.Label(date_frame, text="End Date:", font=("Helvetica", 11), bg="white").pack(side=tk.LEFT, padx=(0, 10))
        
        self.end_date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        self.end_date_entry = tk.Entry(date_frame, textvariable=self.end_date_var, font=("Helvetica", 11), width=12)
        self.end_date_entry.pack(side=tk.LEFT, padx=(0, 20))
        
        # Student filter
        student_frame = tk.Frame(parent, bg="white")
        student_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(student_frame, text="Student Filter:", font=("Helvetica", 11), bg="white").pack(side=tk.LEFT, padx=(0, 10))
        
        self.student_filter_var = tk.StringVar()
        self.student_filter_entry = tk.Entry(student_frame, textvariable=self.student_filter_var, font=("Helvetica", 11), width=30)
        self.student_filter_entry.pack(side=tk.LEFT, padx=(0, 20))
        self.student_filter_entry.insert(0, "Enter name or ID (optional)")
        
        # Buttons frame
        buttons_frame = tk.Frame(parent, bg="white")
        buttons_frame.pack(fill=tk.X)
        
        # Load report button
        load_btn = tk.Button(
            buttons_frame,
            text="Load Report",
            command=self.load_report,
            font=("Helvetica", 10),
            bg="#3498db",
            fg="white",
            relief=tk.FLAT,
            padx=15,
            pady=5
        )
        load_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Refresh button
        refresh_btn = tk.Button(
            buttons_frame,
            text="Refresh",
            command=self.load_recent_attendance,
            font=("Helvetica", 10),
            bg="#f39c12",
            fg="white",
            relief=tk.FLAT,
            padx=15,
            pady=5
        )
        refresh_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Clear filter button
        clear_btn = tk.Button(
            buttons_frame,
            text="Clear Filters",
            command=self.clear_filters,
            font=("Helvetica", 10),
            bg="#95a5a6",
            fg="white",
            relief=tk.FLAT,
            padx=15,
            pady=5
        )
        clear_btn.pack(side=tk.LEFT)
    
    def setup_data_panel(self, parent: tk.Frame):
        """Setup the data display panel."""
        data_frame = tk.LabelFrame(
            parent,
            text="Attendance Data",
            font=("Helvetica", 12, "bold"),
            bg="white",
            fg="#333333",
            padx=20,
            pady=20
        )
        data_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))
        
        # Data treeview
        columns = ("Date", "Time", "Student Name", "Enrollment", "Confidence")
        self.data_tree = ttk.Treeview(
            data_frame,
            columns=columns,
            show="headings",
            height=15
        )
        
        # Setup columns
        column_widths = [100, 80, 200, 120, 100]
        for i, col in enumerate(columns):
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=column_widths[i])
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(data_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack treeview and scrollbar
        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_export_panel(self, parent: tk.Frame):
        """Setup the export panel."""
        export_frame = tk.LabelFrame(
            parent,
            text="Export Options",
            font=("Helvetica", 12, "bold"),
            bg="white",
            fg="#333333",
            padx=20,
            pady=20,
            width=300
        )
        export_frame.pack(side=tk.RIGHT, fill=tk.Y)
        export_frame.pack_propagate(False)
        
        # Export format selection
        tk.Label(
            export_frame,
            text="Export Format:",
            font=("Helvetica", 11, "bold"),
            bg="white"
        ).pack(anchor=tk.W, pady=(0, 10))
        
        self.export_format_var = tk.StringVar(value="CSV")
        formats = ["CSV", "Excel", "JSON"]
        
        for fmt in formats:
            radio = tk.Radiobutton(
                export_frame,
                text=fmt,
                variable=self.export_format_var,
                value=fmt,
                font=("Helvetica", 10),
                bg="white"
            )
            radio.pack(anchor=tk.W, pady=(0, 5))
        
        # Export buttons
        export_btn = tk.Button(
            export_frame,
            text="Export Report",
            command=self.export_report,
            font=("Helvetica", 10),
            bg="#2ecc71",
            fg="white",
            relief=tk.FLAT,
            padx=20,
            pady=10
        )
        export_btn.pack(pady=(20, 10))
        
        export_selected_btn = tk.Button(
            export_frame,
            text="Export Selected",
            command=self.export_selected,
            font=("Helvetica", 10),
            bg="#9b59b6",
            fg="white",
            relief=tk.FLAT,
            padx=20,
            pady=10
        )
        export_selected_btn.pack(pady=(0, 10))
        
        # Statistics
        stats_frame = tk.LabelFrame(
            export_frame,
            text="Statistics",
            font=("Helvetica", 10, "bold"),
            bg="white",
            fg="#333333",
            padx=10,
            pady=10
        )
        stats_frame.pack(fill=tk.X, pady=(20, 0))
        
        self.stats_label = tk.Label(
            stats_frame,
            text="Total Records: 0\nUnique Students: 0",
            font=("Helvetica", 10),
            bg="white",
            justify=tk.LEFT
        )
        self.stats_label.pack(anchor=tk.W)
    
    def load_recent_attendance(self):
        """Load recent attendance data."""
        try:
            # Get recent data (last 7 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            self.load_attendance_data(start_date, end_date, "")
        
        except Exception as e:
            logger.error(f"Error loading recent attendance: {str(e)}")
            messagebox.showerror("Error", f"Failed to load recent attendance: {str(e)}")
    
    def load_report(self):
        """Load report with current filters."""
        try:
            # Parse dates
            start_date_str = self.start_date_var.get()
            end_date_str = self.end_date_var.get()
            student_filter = self.student_filter_var.get()
            
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
            
            if start_date > end_date:
                messagebox.showerror("Error", "Start date cannot be after end date")
                return
            
            self.load_attendance_data(start_date, end_date, student_filter)
        
        except ValueError as e:
            messagebox.showerror("Error", "Invalid date format. Use YYYY-MM-DD")
        except Exception as e:
            logger.error(f"Error loading report: {str(e)}")
            messagebox.showerror("Error", f"Failed to load report: {str(e)}")
    
    def load_attendance_data(self, start_date: datetime, end_date: datetime, student_filter: str):
        """Load attendance data from the database."""
        try:
            # Show loading message
            self.data_tree.delete(*self.data_tree.get_children())
            
            # Get data from database
            if self.attendance_system.db_manager:
                logs = self.attendance_system.db_manager.get_attendance_logs(
                    start_date=start_date,
                    end_date=end_date
                )
            else:
                logs = []
            
            # Filter by student name/ID if provided
            if student_filter.strip():
                filtered_logs = []
                for log in logs:
                    # log format: (id, student_id, timestamp, confidence, name, enrollment_number)
                    if (student_filter.lower() in log[4].lower() or  # name
                        student_filter.lower() in str(log[1]).lower()):  # student_id
                        filtered_logs.append(log)
                logs = filtered_logs
            
            # Store current data
            self.current_report_data = logs
            
            # Display data in treeview
            for log in logs:
                # Parse timestamp
                timestamp = datetime.fromisoformat(log[2].replace('Z', '+00:00'))
                date_str = timestamp.strftime("%Y-%m-%d")
                time_str = timestamp.strftime("%H:%M:%S")
                
                self.data_tree.insert(
                    "",
                    tk.END,
                    values=(
                        date_str,
                        time_str,
                        log[4],  # name
                        f"ID: {log[1]}",  # student_id
                        f"{log[3]:.2f}"  # confidence
                    )
                )
            
            # Update statistics
            self.update_statistics()
            
            logger.info(f"Loaded {len(logs)} attendance records")
        
        except Exception as e:
            logger.error(f"Error loading attendance data: {str(e)}")
            messagebox.showerror("Error", f"Failed to load attendance data: {str(e)}")
    
    def update_statistics(self):
        """Update the statistics display."""
        try:
            total_records = len(self.current_report_data)
            unique_students = len(set(log[1] for log in self.current_report_data))  # student_id
            
            stats_text = f"Total Records: {total_records}\nUnique Students: {unique_students}"
            self.stats_label.config(text=stats_text)
        
        except Exception as e:
            logger.error(f"Error updating statistics: {str(e)}")
    
    def clear_filters(self):
        """Clear all filters and reload recent data."""
        try:
            self.start_date_var.set((datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"))
            self.end_date_var.set(datetime.now().strftime("%Y-%m-%d"))
            self.student_filter_var.set("")
            
            self.load_recent_attendance()
        
        except Exception as e:
            logger.error(f"Error clearing filters: {str(e)}")
    
    def export_report(self):
        """Export the current report."""
        try:
            if not self.current_report_data:
                messagebox.showwarning("Warning", "No data to export")
                return
            
            # Get export format
            format_choice = self.export_format_var.get()
            
            # Get file path
            file_types = {
                "CSV": [("CSV files", "*.csv")],
                "Excel": [("Excel files", "*.xlsx")],
                "JSON": [("JSON files", "*.json")]
            }
            
            default_extension = {
                "CSV": ".csv",
                "Excel": ".xlsx", 
                "JSON": ".json"
            }
            
            file_path = filedialog.asksaveasfilename(
                title="Save Report",
                defaultextension=default_extension[format_choice],
                filetypes=file_types[format_choice]
            )
            
            if not file_path:
                return
            
            # Prepare data
            data = []
            for log in self.current_report_data:
                timestamp = datetime.fromisoformat(log[2].replace('Z', '+00:00'))
                data.append({
                    'Date': timestamp.strftime("%Y-%m-%d"),
                    'Time': timestamp.strftime("%H:%M:%S"),
                    'Student Name': log[4],
                    'Student ID': log[1],
                    'Enrollment Number': log[5],
                    'Confidence': log[3]
                })
            
            # Export data
            if format_choice == "CSV":
                self.export_to_csv(data, file_path)
            elif format_choice == "Excel":
                self.export_to_excel(data, file_path)
            elif format_choice == "JSON":
                self.export_to_json(data, file_path)
            
            messagebox.showinfo("Success", f"Report exported successfully to:\n{file_path}")
        
        except Exception as e:
            logger.error(f"Error exporting report: {str(e)}")
            messagebox.showerror("Error", f"Failed to export report: {str(e)}")
    
    def export_selected(self):
        """Export selected records only."""
        try:
            # Get selected items
            selected_items = self.data_tree.selection()
            if not selected_items:
                messagebox.showwarning("Warning", "No records selected")
                return
            
            # Get data for selected items
            selected_data = []
            for item in selected_items:
                values = self.data_tree.item(item)['values']
                # Convert treeview values back to database format
                selected_data.append({
                    'Date': values[0],
                    'Time': values[1],
                    'Student Name': values[2],
                    'Student ID': values[3].replace('ID: ', ''),
                    'Confidence': float(values[4])
                })
            
            # Export selected data
            format_choice = self.export_format_var.get()
            file_types = {
                "CSV": [("CSV files", "*.csv")],
                "Excel": [("Excel files", "*.xlsx")],
                "JSON": [("JSON files", "*.json")]
            }
            
            file_path = filedialog.asksaveasfilename(
                title="Save Selected Records",
                defaultextension=".csv",
                filetypes=file_types[format_choice]
            )
            
            if not file_path:
                return
            
            if format_choice == "CSV":
                self.export_to_csv(selected_data, file_path)
            elif format_choice == "Excel":
                self.export_to_excel(selected_data, file_path)
            elif format_choice == "JSON":
                self.export_to_json(selected_data, file_path)
            
            messagebox.showinfo("Success", f"Selected records exported successfully to:\n{file_path}")
        
        except Exception as e:
            logger.error(f"Error exporting selected records: {str(e)}")
            messagebox.showerror("Error", f"Failed to export selected records: {str(e)}")
    
    def export_to_csv(self, data: List[Dict], file_path: str):
        """Export data to CSV format."""
        try:
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            logger.info(f"Exported {len(data)} records to CSV: {file_path}")
        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            raise
    
    def export_to_excel(self, data: List[Dict], file_path: str):
        """Export data to Excel format."""
        try:
            df = pd.DataFrame(data)
            df.to_excel(file_path, index=False, engine='openpyxl')
            logger.info(f"Exported {len(data)} records to Excel: {file_path}")
        except Exception as e:
            logger.error(f"Error exporting to Excel: {str(e)}")
            raise
    
    def export_to_json(self, data: List[Dict], file_path: str):
        """Export data to JSON format."""
        try:
            import json
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Exported {len(data)} records to JSON: {file_path}")
        except Exception as e:
            logger.error(f"Error exporting to JSON: {str(e)}")
            raise