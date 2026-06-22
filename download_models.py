"""
Download the required face detection model files.
Run this once before using the face scanner.
"""

import urllib.request
import os

MODELS_DIR = "models"

FILES = {
    "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
}


def download_models():
    """Download face detection DNN model files."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    for filename, url in FILES.items():
        filepath = os.path.join(MODELS_DIR, filename)

        if os.path.exists(filepath):
            print(f"  Already exists: {filename}")
            continue

        print(f"  Downloading: {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"  Done: {filename} ({os.path.getsize(filepath) / 1024 / 1024:.1f} MB)")
        except Exception as e:
            print(f"  ERROR downloading {filename}: {e}")
            return False

    print("\nAll models downloaded successfully!")
    return True


if __name__ == "__main__":
    print("Downloading face detection models...\n")
    download_models()
