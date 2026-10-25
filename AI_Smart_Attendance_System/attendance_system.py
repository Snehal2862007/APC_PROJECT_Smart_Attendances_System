"""
AI-Based Smart Attendance System with Security Features

Features:
- Real-time face detection and recognition
- Euclidean distance matching with 0.6 threshold
- Visual bounding box overlay (green=authorized, red=unknown)
- Anti-duplication verification
- Automated CSV attendance logging
- Frame pipeline optimization (25% downsample, BGR to RGB)
"""

import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
import os
from encode_faces import load_encodings, encode_authorized_faces


class AttendanceSystem:
    def __init__(self, encoding_file="encodings.pkl", csv_file="attendance_records.csv"):
        """
        Initialize the attendance system.
        
        Args:
            encoding_file (str): Path to face encodings file
            csv_file (str): Path to attendance CSV file
        """
        self.encoding_file = encoding_file
        self.csv_file = csv_file
        self.encodings = {}
        self.known_face_names = []
        self.known_face_encodings = []
        self.tolerance = 0.6  # Euclidean distance threshold
        self.frame_scale = 0.25  # Downsample to 25% for performance
        
        # Initialize CSV file
        self._init_csv()
        
        # Load face encodings
        self._load_encodings()
    
    def _init_csv(self):
        """Initialize CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_file):
            df = pd.DataFrame(columns=['Name', 'Date', 'Timestamp', 'Status'])
            df.to_csv(self.csv_file, index=False)
            print(f"[INFO] Created attendance log: {self.csv_file}")
    
    def _load_encodings(self):
        """Load face encodings from file or generate new ones."""
        # Try to load existing encodings
        self.encodings = load_encodings(self.encoding_file)
        
        if not self.encodings:
            print("[INFO] No encodings found. Generating new ones...")
            self.encodings = encode_authorized_faces(encoding_file=self.encoding_file)
        
        # Convert to lists for face_recognition
        self.known_face_names = list(self.encodings.keys())
        self.known_face_encodings = list(self.encodings.values())
        
        print(f"[INFO] Loaded {len(self.known_face_names)} authorized face(s)")
    
    def _is_already_marked(self, name):
        """
        Check if attendance for this person is already marked today.
        Anti-duplication verification using Pandas.
        
        Args:
            name (str): Person's name
        
        Returns:
            bool: True if already marked, False otherwise
        """
        if not os.path.exists(self.csv_file):
            return False
        
        try:
            df = pd.read_csv(self.csv_file)
            today = datetime.now().strftime('%Y-%m-%d')
            
            # Check if person already marked today
            marked = df[(df['Name'] == name) & (df['Date'] == today)]
            return len(marked) > 0
        except Exception as e:
            print(f"[ERROR] Failed to check attendance: {str(e)}")
            return False
    
    def _log_attendance(self, name, status="Authorized"):
        """
        Log attendance to CSV file.
        
        Args:
            name (str): Person's name
            status (str): Authorization status
        """
        try:
            now = datetime.now()
            date_str = now.strftime('%Y-%m-%d')
            time_str = now.strftime('%H:%M:%S')
            
            new_entry = {
                'Name': name,
                'Date': date_str,
                'Timestamp': time_str,
                'Status': status
            }
            
            df = pd.read_csv(self.csv_file)
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
            df.to_csv(self.csv_file, index=False)
            
            print(f"[LOG] Attendance marked: {name} at {time_str}")
            
        except Exception as e:
            print(f"[ERROR] Failed to log attendance: {str(e)}")
    
    def _match_face(self, face_encoding):
        """
        Match face encoding against known encodings using Euclidean distance.
        
        Args:
            face_encoding (numpy.ndarray): 128-dimensional face encoding
        
        Returns:
            tuple: (name, distance) or ("Unknown", distance)
        """
        if not self.known_face_encodings:
            return "Unknown", 1.0
        
        # Calculate Euclidean distances
        distances = face_recognition.face_distance(
            self.known_face_encodings, 
            face_encoding
        )
        
        # Find minimum distance
        min_distance = np.min(distances)
        min_index = np.argmin(distances)
        
        # Check if within tolerance threshold
        if min_distance <= self.tolerance:
            return self.known_face_names[min_index], min_distance
        else:
            return "Unknown", min_distance
    
    def process_frame(self, frame):
        """
        Process a single video frame for face detection and recognition.
        Includes frame pipeline optimization and visual overlay.
        
        Args:
            frame (numpy.ndarray): Input video frame
        
        Returns:
            numpy.ndarray: Frame with bounding box overlays
        """
        # Frame Pipeline Optimization:
        # 1. Downsample to 25% scale to minimize latency
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_scale, fy=self.frame_scale)
        
        # 2. Convert BGR to RGB color schema
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect face locations in the small frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        # Process each detected face
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale coordinates back to original frame size
            top = int(top / self.frame_scale)
            right = int(right / self.frame_scale)
            bottom = int(bottom / self.frame_scale)
            left = int(left / self.frame_scale)
            
            # Match face using Euclidean distance
            name, distance = self._match_face(face_encoding)
            
            # Determine color based on authorization
            if name == "Unknown":
                color = (0, 0, 255)  # Red for unknown threat
                status = "Unauthorized"
            else:
                color = (0, 255, 0)  # Green for authorized
                status = "Authorized"
                
                # Anti-duplication check
                if not self._is_already_marked(name):
                    self._log_attendance(name, status)
            
            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label with name and distance
            label = f"{name} ({distance:.2f})"
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 6),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def run(self, camera_index=0):
        """
        Run the attendance system with live camera feed.
        
        Args:
            camera_index (int): Camera device index (default: 0)
        """
        print("[INFO] Starting attendance system...")
        print("[INFO] Press 'q' to quit")
        
        # Initialize camera
        video_capture = cv2.VideoCapture(camera_index)
        
        if not video_capture.isOpened():
            print("[ERROR] Failed to open camera")
            return
        
        print("[INFO] Camera initialized successfully")
        
        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            
            if not ret:
                print("[ERROR] Failed to capture frame")
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Display the resulting frame
            cv2.imshow('AI Smart Attendance System', processed_frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        video_capture.release()
        cv2.destroyAllWindows()
        print("[INFO] Attendance system stopped")


def main():
    """Main entry point for the attendance system."""
    print("=" * 60)
    print("AI-Based Smart Attendance System")
    print("=" * 60)
    
    # Initialize system
    system = AttendanceSystem()
    
    if not system.known_face_names:
        print("\n[ERROR] No authorized faces loaded!")
        print("[INFO] Please add face images to 'authorized_faces/' directory")
        print("[INFO] Then run: python encode_faces.py")
        return
    
    # Run attendance system
    system.run()


if __name__ == "__main__":
    main()
