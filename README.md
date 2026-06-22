# Face Scanner Verification System

A Python-based face verification system that registers faces into a database
and verifies identity for access control.

## Features

- **Register** — Scan a face and store it in the database with a name
- **Verify** — Scan a face to check if it's registered (grants/denies access)
- **Duplicate prevention** — Same face cannot be registered twice
- **Local database** — Uses SQLite, no external server needed
- **Real-time webcam** — Live camera feed with face detection overlay
- **No C++ compiler needed** — Uses pure OpenCV (no dlib)

## Setup

Dependencies are already installed. If you need to reinstall:

```bash
py -m pip install opencv-python opencv-contrib-python numpy
```

Download models (already done):
```bash
py download_models.py
```

## Run

```bash
py face_scanner.py
```

## How it works

1. **Detection**: Uses OpenCV's DNN module with a pre-trained Caffe model
   to detect faces in real-time from the webcam.

2. **Registration**: Captures the face region, normalizes it, creates a
   128x128 grayscale encoding, and stores it in SQLite.

3. **Verification**: Compares a captured face encoding against all stored
   encodings using cosine similarity. If the match exceeds the threshold,
   access is granted.

## Controls

- **SPACE** — Capture/scan face
- **ESC** — Cancel current operation

## Files

- `face_scanner.py` — Main application (menu, register, verify)
- `database.py` — SQLite database operations
- `download_models.py` — Downloads DNN model files
- `models/` — Face detection model files
- `face_database.db` — Auto-created database (stores face data)
