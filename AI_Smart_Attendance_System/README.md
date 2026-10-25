# AI-Based Smart Attendance System

An intelligent attendance tracking system using face recognition technology with security features and automated logging.

## Features

- **Real-time Face Detection**: Captures and processes video streams from webcam
- **128-Dimensional Face Encoding**: Converts face images into unique feature vectors
- **Euclidean Distance Matching**: Compares faces with configurable tolerance threshold (0.6)
- **Visual Security Overlay**: 
  - Green bounding box for authorized personnel
  - Red bounding box for unknown/unauthorized individuals
- **Anti-Duplication Verification**: Prevents multiple attendance entries for the same person on the same day
- **Automated CSV Logging**: Records attendance with name, date, timestamp, and status
- **Frame Pipeline Optimization**: 
  - 25% frame downsampling for reduced latency
  - BGR to RGB color space conversion
- **Heuristic Threshold Security**: Configurable distance tolerance for authorization

## System Requirements

### Operating System
- Windows 10/11
- macOS
- Linux (Ubuntu recommended)

### Programming Language
- Python 3.8 to Python 3.11

### Hardware
- Built-in laptop webcam or external USB camera

## Installation

### Step 1: Clone or Download the Project

```bash
cd AI_Smart_Attendance_System
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The `dlib` library requires CMake to be installed on your system.

#### Installing CMake (if needed)

**Windows:**
```bash
# Download and install CMake from https://cmake.org/download/
# Or use chocolatey:
choco install cmake
```

**macOS:**
```bash
brew install cmake
```

**Linux (Ubuntu):**
```bash
sudo apt-get update
sudo apt-get install cmake
```

### Step 3: Add Authorized Face Images

Place face images of authorized personnel in the `authorized_faces/` directory:

- Name images after the person (e.g., `alice.jpg`, `bob.jpg`, `charlie.png`)
- Supported formats: JPG, JPEG, PNG
- Ensure clear, front-facing photos with good lighting

Example structure:
```
AI_Smart_Attendance_System/
├── authorized_faces/
│   ├── alice.jpg
│   ├── bob.jpg
│   └── charlie.png
├── attendance_system.py
├── encode_faces.py
├── requirements.txt
└── README.md
```

### Step 4: Encode Faces

Generate 128-dimensional face encodings from the authorized images:

```bash
python encode_faces.py
```

This will:
- Load all images from `authorized_faces/`
- Detect faces in each image
- Generate encodings
- Save to `encodings.pkl` file

## Usage

### Running the Attendance System

```bash
python attendance_system.py
```

The system will:
1. Load pre-computed face encodings
2. Initialize the camera
3. Start real-time face detection
4. Display bounding boxes around detected faces
5. Log attendance automatically for authorized personnel
6. Press `q` to quit

### Using a Different Camera

To use a camera other than the default (index 0), modify the `camera_index` parameter:

```python
# In attendance_system.py, change:
system.run(camera_index=0)  # Default camera

# To use a different camera:
system.run(camera_index=1)  # Second camera
```

## How It Works

### 1. Template Matrix Compilation
The system loops through the `authorized_faces/` directory and transforms each image into a 128-dimensional facial feature vector stored in memory.

### 2. Frame Pipeline Optimization
- Captures video stream from camera
- Downsamples frames to 25% scale for performance
- Converts pixel values from BGR to RGB color schema

### 3. Euclidean Distance Mapping
- Tracks facial coordinates across video streams
- Computes numerical distance vectors against registered user matrices
- Identifies the match with the lowest variance

### 4. Heuristic Threshold Security
- Distances below 0.6 tolerance = Authorized (green box)
- Distances above 0.6 tolerance = Unknown threat (red box)

### 5. Anti-Duplication Verification
- Uses Pandas to filter existing attendance records
- Prevents redundant time clock logging for the same person on the same date

### 6. Automated CSV Persistence
- Appends new entries with: name, date, timestamp, verification status
- Immediately flushes modifications to physical file storage

## Output Files

### encodings.pkl
Binary file containing 128-dimensional face encodings for all authorized personnel.

### attendance_records.csv
CSV file with attendance records:
```
Name,Date,Timestamp,Status
alice,2026-10-08,09:15:23,Authorized
bob,2026-10-08,09:20:45,Authorized
```

## Configuration

### Adjusting Tolerance Threshold

Modify the `tolerance` value in `attendance_system.py`:

```python
self.tolerance = 0.6  # Default: 0.6
```

- Lower values (e.g., 0.5) = More strict matching (higher false negatives)
- Higher values (e.g., 0.7) = More lenient matching (higher false positives)

### Adjusting Frame Scale

Modify the `frame_scale` value for performance tuning:

```python
self.frame_scale = 0.25  # Default: 25% scale
```

- Lower values = Faster processing but less accuracy
- Higher values = Slower processing but more accuracy

## Troubleshooting

### Camera Not Opening
- Check if camera is connected and not in use by another application
- Try different camera indices (0, 1, 2, etc.)

### No Faces Detected
- Ensure good lighting conditions
- Position camera at eye level
- Check if face images in `authorized_faces/` are clear and front-facing

### dlib Installation Error
- Ensure CMake is installed
- Try installing Visual Studio Build Tools (Windows)
- On Linux: `sudo apt-get install build-essential`

### Multiple Faces in Single Image
The system uses the first detected face. Ensure each image in `authorized_faces/` contains only one face.

## Security Considerations

- The 0.6 tolerance threshold balances security and usability
- Unknown faces are marked with red bounding boxes for visual alert
- All attendance is logged with timestamps for audit trails
- Encodings are stored locally (no cloud transmission)

## License

This project is provided as-is for educational and research purposes.

## Credits

Built with:
- OpenCV (opencv-python)
- face_recognition library
- NumPy
- Pandas
