"""
Face Scanner - MySQL Attendance Gate System

Handles face detection, LBP encoding extraction, and check-in/check-out updates.
"""

import os
import base64
import hashlib
import numpy as np
import cv2
from functools import wraps
from flask import Flask, request, jsonify, session
from flask_cors import CORS

from database import init_db, save_user, get_all_users, delete_user, get_user_count, mark_attendance

# Initialize database tables
init_db()

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)  # Allow cross-origin requests from the React dev client

# Face detection config
MODELS_DIR = "models"
PROTOTXT = os.path.join(MODELS_DIR, "deploy.prototxt")
CAFFEMODEL = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
DETECTION_CONFIDENCE = 0.75
# Chi-squared distance threshold: LOWER distance = more similar faces
# Same person typically scores 10-30, different person scores 80+
MATCH_THRESHOLD = 18

net = None
eye_cascade = None


def get_net():
    global net
    if net is None:
        net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
    return net


def get_eye_cascade():
    """Load the Haar cascade for eye detection."""
    global eye_cascade
    if eye_cascade is None:
        eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    return eye_cascade


def detect_eyes(face_gray):
    """
    Detect eyes in a grayscale face region.
    """
    cascade = get_eye_cascade()
    h = face_gray.shape[0]
    upper_face = face_gray[0:int(h * 0.65), :]

    eyes = cascade.detectMultiScale(
        upper_face,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(20, 20),
        maxSize=(int(upper_face.shape[1] * 0.4), int(upper_face.shape[0] * 0.5))
    )
    return eyes


def decode_image(data_url):
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def detect_faces(frame):
    face_net = get_net()
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > DETECTION_CONFIDENCE:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            faces.append({"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)})
    return faces


def get_face_encoding(frame, face):
    """
    Extract face encoding using Local Binary Pattern (LBP) histograms.
    Requires eyes to be visible — rejects if no eyes detected.
    """
    x, y, w, h = face["x"], face["y"], face["w"], face["h"]
    face_roi = frame[y:y + h, x:x + w]
    if face_roi.size == 0:
        return None

    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    # Reject if the region is too bright/washed out or too uniform
    mean_val = gray.mean()
    std_val = gray.std()
    if mean_val > 240 or std_val < 15:
        return None

    # Eye detection — must see at least one eye
    eyes = detect_eyes(gray)
    if len(eyes) < 1:
        return None

    resized = cv2.resize(gray, (160, 160))
    normalized = cv2.equalizeHist(resized)

    # Split face into grid regions and compute LBP histogram per region
    cell_size = 20  # 160 / 8 = 20 pixels per cell
    n_cells = 8
    histograms = []

    for i in range(n_cells):
        for j in range(n_cells):
            cell = normalized[i * cell_size:(i + 1) * cell_size,
                              j * cell_size:(j + 1) * cell_size]

            lbp = compute_lbp(cell)

            hist, _ = np.histogram(lbp.ravel(), bins=32, range=(0, 256))
            hist = hist.astype(np.float32)

            hist_sum = hist.sum()
            if hist_sum > 0:
                hist = hist / hist_sum

            histograms.append(hist)

    encoding = np.concatenate(histograms).astype(np.float32)
    return encoding


def compute_lbp(image):
    """Compute Local Binary Pattern using vectorized numpy."""
    img = image.astype(np.int16)
    center = img[1:-1, 1:-1]
    rows, cols = img.shape
    lbp = np.zeros((rows - 2, cols - 2), dtype=np.uint8)

    lbp |= ((img[0:-2, 0:-2] >= center).astype(np.uint8) << 7)
    lbp |= ((img[0:-2, 1:-1] >= center).astype(np.uint8) << 6)
    lbp |= ((img[0:-2, 2:]   >= center).astype(np.uint8) << 5)
    lbp |= ((img[1:-1, 2:]   >= center).astype(np.uint8) << 4)
    lbp |= ((img[2:,   2:]   >= center).astype(np.uint8) << 3)
    lbp |= ((img[2:,   1:-1] >= center).astype(np.uint8) << 2)
    lbp |= ((img[2:,   0:-2] >= center).astype(np.uint8) << 1)
    lbp |= ((img[1:-1, 0:-2] >= center).astype(np.uint8) << 0)

    return lbp


def compare_faces(enc1, enc2):
    """
    Compare two LBP histogram encodings using chi-squared distance.
    """
    if enc1 is None or enc2 is None:
        return 999.0
    denominator = enc1 + enc2
    mask = denominator > 0
    chi_sq = np.sum(((enc1[mask] - enc2[mask]) ** 2) / denominator[mask])
    return float(chi_sq)


# ==================== API ROUTES ====================


@app.route("/api/verify", methods=["POST"])
def api_verify():
    """Verify a face and mark attendance in MySQL database."""
    data = request.json
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    if get_user_count() == 0:
        return jsonify({"verified": False, "message": "No users registered in database"}), 200

    frame = decode_image(data["image"])
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    faces = detect_faces(frame)
    if len(faces) != 1:
        return jsonify({"verified": False, "message": "no_face"}), 200

    encoding = get_face_encoding(frame, faces[0])
    if encoding is None:
        return jsonify({"verified": False, "message": "no_eyes"}), 200

    users = get_all_users()
    best_distance, best_emp_id, best_name = 999.0, None, None
    for emp_id, name, stored_enc in users:
        distance = compare_faces(encoding, stored_enc)
        if distance < best_distance:
            best_distance, best_emp_id, best_name = distance, emp_id, name

    if best_distance < MATCH_THRESHOLD:
        # Mark attendance automatically in MySQL
        attendance_info = mark_attendance(best_emp_id)
        confidence = max(0, min(100, (1 - best_distance / MATCH_THRESHOLD) * 100))
        return jsonify({
            "verified": True,
            "employee_id": best_emp_id,
            "name": best_name,
            "confidence": round(confidence, 1),
            "attendance": attendance_info
        })

    return jsonify({"verified": False, "message": "not_recognized"})

from database import manual_checkout

@app.route("/api/checkout", methods=["POST"])
def api_checkout():
    """Manually checkout an employee after they click the checkout button."""
    data = request.json
    if not data or "employee_id" not in data:
        return jsonify({"error": "Employee ID required"}), 400
    
    result = manual_checkout(data["employee_id"])
    return jsonify(result)


@app.route("/api/validate_face", methods=["POST"])
def api_validate_face():
    """Validate a face image for quality and eye visibility without saving."""
    data = request.json
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    frame = decode_image(data["image"])
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    faces = detect_faces(frame)
    if len(faces) == 0:
        return jsonify({"error": "No face detected. Adjust lighting and face the camera directly."}), 400
    if len(faces) > 1:
        return jsonify({"error": "Multiple faces detected. Ensure only one person is visible."}), 400

    encoding = get_face_encoding(frame, faces[0])
    if encoding is None:
        return jsonify({"error": "Face quality check failed. Ensure eyes are visible and look directly at camera."}), 400

    return jsonify({"success": True, "message": "Face verified successfully. Ready to register."})


@app.route("/api/register", methods=["POST"])
def api_register():
    """Register a new face linked to an employee ID."""
    data = request.json
    if not data or "image" not in data or "employee_id" not in data:
        return jsonify({"error": "Image and employee_id required"}), 400

    employee_id = data["employee_id"].strip()
    if not employee_id:
        return jsonify({"error": "Employee ID cannot be empty"}), 400

    frame = decode_image(data["image"])
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    faces = detect_faces(frame)
    if len(faces) == 0:
        return jsonify({"error": "No face detected"}), 400
    if len(faces) > 1:
        return jsonify({"error": "Multiple faces detected"}), 400

    encoding = get_face_encoding(frame, faces[0])
    if encoding is None:
        return jsonify({"error": "Could not encode face. Check lighting and visibility of eyes"}), 400

    # Save face template linked to employee_id in MySQL
    success = save_user(employee_id, encoding)
    if not success:
        return jsonify({"error": f"Failed to save face encoding. Verify employee exists and does not already have a registered face."}), 400

    # Save cropped face image for displaying portrait
    faces_dir = os.path.join("static", "faces")
    os.makedirs(faces_dir, exist_ok=True)

    f = faces[0]
    h_frame, w_frame = frame.shape[:2]
    pad_x = int(f["w"] * 0.4)
    pad_y = int(f["h"] * 0.5)
    x1 = max(0, f["x"] - pad_x)
    y1 = max(0, f["y"] - pad_y)
    x2 = min(w_frame, f["x"] + f["w"] + pad_x)
    y2 = min(h_frame, f["y"] + f["h"] + pad_y)
    face_img = frame[y1:y2, x1:x2]

    face_filename = f"{employee_id}.jpg"
    cv2.imwrite(os.path.join(faces_dir, face_filename), face_img)

    return jsonify({"success": True, "message": f"Employee {employee_id} face template registered successfully!"})


@app.route("/api/users", methods=["GET"])
def api_users():
    """Get all registered face profiles."""
    users = get_all_users()
    user_list = []
    for emp_id, name, _ in users:
        face_filename = f"{emp_id}.jpg"
        face_path = f"/static/faces/{face_filename}"
        has_image = os.path.exists(os.path.join("static", "faces", face_filename))
        user_list.append({
            "employee_id": emp_id,
            "name": name,
            "face_image": face_path if has_image else None
        })
    return jsonify({"users": user_list, "count": len(user_list)})


@app.route("/api/users/<employee_id>", methods=["DELETE"])
def api_delete_user(employee_id):
    """Delete employee face registration."""
    if delete_user(employee_id):
        face_filename = f"{employee_id}.jpg"
        face_path = os.path.join("static", "faces", face_filename)
        if os.path.exists(face_path):
            os.remove(face_path)
        return jsonify({"success": True, "message": "Face profile deleted."})
    return jsonify({"error": "Face profile not found"}), 404


if __name__ == "__main__":
    init_db()
    print("\n" + "=" * 50)
    print("   FACE SCANNER - MySQL Attendance Gateway")
    print("=" * 50)
    print(f"\n   Verify Server: http://localhost:5000")
    print(f"   Users Enrolled: {get_user_count()}")
    print("=" * 50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
