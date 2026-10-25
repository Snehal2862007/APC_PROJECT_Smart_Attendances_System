"""
Face Encoding Module
Compiles 128-dimensional facial feature vectors from authorized face images
and stores them in memory for real-time matching.
"""

import face_recognition
import pickle
import os
from pathlib import Path

def encode_authorized_faces(faces_dir="authorized_faces", encoding_file="encodings.pkl"):
    """
    Loop through authorized_faces directory, encode each face image into
    128-dimensional vectors, and save to pickle file.
    
    Args:
        faces_dir (str): Directory containing authorized face images
        encoding_file (str): Output file to store encodings
    
    Returns:
        dict: Dictionary of {name: encoding} pairs
    """
    print(f"[INFO] Loading authorized faces from '{faces_dir}' directory...")
    
    # Initialize encodings dictionary
    encodings = {}
    
    # Check if directory exists
    if not os.path.exists(faces_dir):
        print(f"[ERROR] Directory '{faces_dir}' not found. Creating it...")
        os.makedirs(faces_dir)
        print(f"[INFO] Please add face images (e.g., alice.jpg, bob.jpg) to '{faces_dir}' directory.")
        return encodings
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(Path(faces_dir).glob(ext))
    
    if not image_files:
        print(f"[WARNING] No face images found in '{faces_dir}' directory.")
        print(f"[INFO] Please add face images (e.g., alice.jpg, bob.jpg) to '{faces_dir}' directory.")
        return encodings
    
    print(f"[INFO] Found {len(image_files)} face image(s). Processing...")
    
    # Process each image
    for image_path in image_files:
        # Extract name from filename (without extension)
        name = image_path.stem
        
        try:
            # Load image
            print(f"[PROCESSING] Encoding {name}...")
            image = face_recognition.load_image_file(str(image_path))
            
            # Detect faces in image
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                print(f"[WARNING] No face detected in {image_path.name}")
                continue
            
            if len(face_locations) > 1:
                print(f"[WARNING] Multiple faces detected in {image_path.name}. Using first face.")
            
            # Encode face (128-dimensional vector)
            face_encoding = face_recognition.face_encodings(image, face_locations)[0]
            
            # Store encoding
            encodings[name] = face_encoding
            print(f"[SUCCESS] Encoded {name} successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to encode {image_path.name}: {str(e)}")
    
    # Save encodings to pickle file
    if encodings:
        with open(encoding_file, 'wb') as f:
            pickle.dump(encodings, f)
        print(f"[SUCCESS] Saved {len(encodings)} encoding(s) to '{encoding_file}'")
    else:
        print(f"[WARNING] No encodings generated.")
    
    return encodings


def load_encodings(encoding_file="encodings.pkl"):
    """
    Load pre-computed face encodings from pickle file.
    
    Args:
        encoding_file (str): Path to encoding file
    
    Returns:
        dict: Dictionary of {name: encoding} pairs
    """
    if not os.path.exists(encoding_file):
        print(f"[WARNING] Encoding file '{encoding_file}' not found.")
        return {}
    
    try:
        with open(encoding_file, 'rb') as f:
            encodings = pickle.load(f)
        print(f"[INFO] Loaded {len(encodings)} encoding(s) from '{encoding_file}'")
        return encodings
    except Exception as e:
        print(f"[ERROR] Failed to load encodings: {str(e)}")
        return {}


if __name__ == "__main__":
    # Encode faces when run directly
    encodings = encode_authorized_faces()
    
    if encodings:
        print("\n[SUMMARY] Successfully encoded the following users:")
        for name in encodings.keys():
            print(f"  - {name}")
    else:
        print("\n[SUMMARY] No faces encoded. Please add images to authorized_faces/ directory.")
