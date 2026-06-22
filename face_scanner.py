"""
Face Scanner Verification System - Full GUI (no terminal input needed)

All interactions happen in the OpenCV window:
- On-screen keyboard for name input
- Visual menu, registration, and verification
- No terminal required for any operation
"""

import sys
import os
import time
import cv2
import numpy as np

from database import init_db, save_user, get_all_users, delete_user, get_user_count
from download_models import download_models

# Confidence threshold for face matching
MATCH_THRESHOLD = 80

MODELS_DIR = "models"
PROTOTXT = os.path.join(MODELS_DIR, "deploy.prototxt")
CAFFEMODEL = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
DETECTION_CONFIDENCE = 0.6

# UI Colors (BGR)
COLOR_GREEN = (0, 220, 80)
COLOR_RED = (0, 0, 220)
COLOR_BLUE = (220, 160, 0)
COLOR_YELLOW = (0, 220, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_DARK_BG = (40, 40, 40)
COLOR_PANEL_BG = (30, 30, 30)
COLOR_GRAY = (150, 150, 150)


def load_face_detector():
    """Load the DNN-based face detector."""
    if not os.path.exists(PROTOTXT) or not os.path.exists(CAFFEMODEL):
        print("Downloading face detection models...")
        if not download_models():
            sys.exit(1)
    return cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)


def detect_faces(frame, net):
    """Detect faces using OpenCV DNN."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > DETECTION_CONFIDENCE:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces


def get_face_encoding(frame, face_box):
    """Extract and encode a face region."""
    x, y, w, h = face_box
    face_roi = frame[y:y + h, x:x + w]
    if face_roi.size == 0:
        return None

    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(gray_face, (128, 128))
    face_normalized = cv2.equalizeHist(face_resized)
    encoding = face_normalized.flatten().astype(np.float32)

    norm = np.linalg.norm(encoding)
    if norm > 0:
        encoding = encoding / norm
    return encoding


def compare_faces(enc1, enc2):
    """Compare two face encodings using cosine similarity."""
    if enc1 is None or enc2 is None:
        return 0
    return max(0, float(np.dot(enc1, enc2)) * 100)


def is_face_already_registered(face_encoding):
    """Check if face is already in database."""
    users = get_all_users()
    if not users:
        return False, None, 0

    best_score, best_name = 0, None
    for _, name, stored_enc in users:
        score = compare_faces(face_encoding, stored_enc)
        if score > best_score:
            best_score, best_name = score, name

    if best_score > MATCH_THRESHOLD:
        return True, best_name, best_score
    return False, None, 0


def verify_face(face_encoding):
    """Verify a face against the database."""
    users = get_all_users()
    if not users:
        return False, None, 0

    best_score, best_name = 0, None
    for _, name, stored_enc in users:
        score = compare_faces(face_encoding, stored_enc)
        if score > best_score:
            best_score, best_name = score, name

    if best_score > MATCH_THRESHOLD:
        return True, best_name, best_score
    return False, None, 0


# ==================== UI DRAWING ====================


def draw_face_frame(display, x, y, w, h, color, thickness=2):
    """Draw corner brackets around a face."""
    cl = min(w, h) // 4
    cv2.line(display, (x, y), (x + cl, y), color, thickness)
    cv2.line(display, (x, y), (x, y + cl), color, thickness)
    cv2.line(display, (x + w, y), (x + w - cl, y), color, thickness)
    cv2.line(display, (x + w, y), (x + w, y + cl), color, thickness)
    cv2.line(display, (x, y + h), (x + cl, y + h), color, thickness)
    cv2.line(display, (x, y + h), (x, y + h - cl), color, thickness)
    cv2.line(display, (x + w, y + h), (x + w - cl, y + h), color, thickness)
    cv2.line(display, (x + w, y + h), (x + w, y + h - cl), color, thickness)


def draw_top_bar(display, text, color):
    """Draw top status bar."""
    h, w = display.shape[:2]
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), COLOR_DARK_BG, -1)
    cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
    cv2.circle(display, (20, 25), 8, color, -1)
    cv2.putText(display, text, (38, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
    time_str = time.strftime("%H:%M:%S")
    cv2.putText(display, time_str, (w - 100, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GRAY, 1)


def draw_bottom_bar(display, text):
    """Draw bottom instruction bar."""
    h, w = display.shape[:2]
    overlay = display.copy()
    cv2.rectangle(overlay, (0, h - 45), (w, h), COLOR_DARK_BG, -1)
    cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
    cv2.putText(display, text, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def draw_center_message(display, message, color, y_offset=0):
    """Draw a centered message with dark background."""
    h, w = display.shape[:2]
    font_scale = 0.7
    thickness = 2
    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = h - 70 + y_offset

    overlay = display.copy()
    pad = 12
    cv2.rectangle(overlay, (text_x - pad, text_y - text_size[1] - pad),
                  (text_x + text_size[0] + pad, text_y + pad), COLOR_DARK_BG, -1)
    cv2.addWeighted(overlay, 0.8, display, 0.2, 0, display)
    cv2.putText(display, message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def draw_result_screen(display, granted, name="", confidence=0):
    """Draw access granted/denied result screen."""
    h, w = display.shape[:2]
    overlay = display.copy()

    if granted:
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 80, 0), -1)
        cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)

        cy = h // 2 - 30
        cv2.circle(display, (w // 2, cy - 40), 50, COLOR_GREEN, 3)
        cv2.line(display, (w // 2 - 20, cy - 40), (w // 2 - 5, cy - 25), COLOR_GREEN, 3)
        cv2.line(display, (w // 2 - 5, cy - 25), (w // 2 + 25, cy - 60), COLOR_GREEN, 3)

        text = "ACCESS GRANTED"
        ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        cv2.putText(display, text, ((w - ts[0]) // 2, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_GREEN, 3)

        welcome = f"Welcome, {name}!"
        ws = cv2.getTextSize(welcome, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.putText(display, welcome, ((w - ws[0]) // 2, cy + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)

        # Confidence bar
        bar_w, bar_x, bar_y = 200, (w - 200) // 2, cy + 100
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), (80, 80, 80), -1)
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + int(bar_w * confidence / 100), bar_y + 20), COLOR_GREEN, -1)
        cv2.putText(display, f"Match: {confidence:.1f}%", (bar_x, bar_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1)
    else:
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 80), -1)
        cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)

        cy = h // 2 - 20
        cv2.circle(display, (w // 2, cy - 40), 50, COLOR_RED, 3)
        cv2.line(display, (w // 2 - 20, cy - 60), (w // 2 + 20, cy - 20), COLOR_RED, 3)
        cv2.line(display, (w // 2 + 20, cy - 60), (w // 2 - 20, cy - 20), COLOR_RED, 3)

        text = "ACCESS DENIED"
        ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        cv2.putText(display, text, ((w - ts[0]) // 2, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_RED, 3)

        sub = "Face not recognized"
        ss = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(display, sub, ((w - ss[0]) // 2, cy + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GRAY, 2)

    return display


# ==================== ON-SCREEN KEYBOARD ====================


class OnScreenKeyboard:
    """An on-screen keyboard drawn on the OpenCV window for text input."""

    KEYS_LAYOUT = [
        list("QWERTYUIOP"),
        list("ASDFGHJKL"),
        list("ZXCVBNM"),
    ]

    def __init__(self):
        self.text = ""
        self.active = False
        self.title = "Enter Name"
        self.key_w = 45
        self.key_h = 45
        self.gap = 5
        self.start_y = 200

    def reset(self, title="Enter Name"):
        self.text = ""
        self.active = True
        self.title = title

    def draw(self, display):
        """Draw the keyboard and text input on the frame."""
        h, w = display.shape[:2]

        # Dark overlay
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)

        # Title
        ts = cv2.getTextSize(self.title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.putText(display, self.title, ((w - ts[0]) // 2, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)

        # Text input box
        box_x, box_y = 50, 70
        box_w = w - 100
        cv2.rectangle(display, (box_x, box_y), (box_x + box_w, box_y + 50), (60, 60, 60), -1)
        cv2.rectangle(display, (box_x, box_y), (box_x + box_w, box_y + 50), COLOR_GREEN, 2)

        # Display typed text with cursor
        display_text = self.text + "|"
        cv2.putText(display, display_text, (box_x + 15, box_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WHITE, 2)

        # Draw keyboard rows
        total_keyboard_w = len(self.KEYS_LAYOUT[0]) * (self.key_w + self.gap)
        start_x_base = (w - total_keyboard_w) // 2

        for row_idx, row in enumerate(self.KEYS_LAYOUT):
            row_w = len(row) * (self.key_w + self.gap)
            start_x = (w - row_w) // 2
            y = self.start_y + row_idx * (self.key_h + self.gap)

            for col_idx, key in enumerate(row):
                x = start_x + col_idx * (self.key_w + self.gap)
                cv2.rectangle(display, (x, y), (x + self.key_w, y + self.key_h), (70, 70, 70), -1)
                cv2.rectangle(display, (x, y), (x + self.key_w, y + self.key_h), (120, 120, 120), 1)

                key_ts = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                key_x = x + (self.key_w - key_ts[0]) // 2
                key_y = y + (self.key_h + key_ts[1]) // 2
                cv2.putText(display, key, (key_x, key_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)

        # Special keys row
        special_y = self.start_y + 3 * (self.key_h + self.gap)

        # Space bar
        space_w = 250
        space_x = (w - space_w) // 2
        cv2.rectangle(display, (space_x, special_y), (space_x + space_w, special_y + self.key_h), (70, 70, 70), -1)
        cv2.rectangle(display, (space_x, special_y), (space_x + space_w, special_y + self.key_h), (120, 120, 120), 1)
        sp_ts = cv2.getTextSize("SPACE", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(display, "SPACE", (space_x + (space_w - sp_ts[0]) // 2, special_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1)

        # Backspace
        bk_x = space_x + space_w + 15
        bk_w = 80
        cv2.rectangle(display, (bk_x, special_y), (bk_x + bk_w, special_y + self.key_h), (70, 70, 70), -1)
        cv2.rectangle(display, (bk_x, special_y), (bk_x + bk_w, special_y + self.key_h), (120, 120, 120), 1)
        cv2.putText(display, "<-", (bk_x + 20, special_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 2)

        # Bottom buttons
        btn_y = special_y + self.key_h + 20

        # CONFIRM button
        confirm_x = w // 2 - 160
        cv2.rectangle(display, (confirm_x, btn_y), (confirm_x + 140, btn_y + 45), COLOR_GREEN, -1)
        cv2.putText(display, "CONFIRM", (confirm_x + 20, btn_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_BLACK, 2)

        # CANCEL button
        cancel_x = w // 2 + 20
        cv2.rectangle(display, (cancel_x, btn_y), (cancel_x + 140, btn_y + 45), COLOR_RED, -1)
        cv2.putText(display, "CANCEL", (cancel_x + 25, btn_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_BLACK, 2)

        # Instructions
        cv2.putText(display, "Type using keyboard keys | ENTER=Confirm | ESC=Cancel | BACKSPACE=Delete",
                    (30, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GRAY, 1)

        return display

    def handle_key(self, key):
        """
        Handle a keypress. Returns:
            'confirm' if user pressed Enter
            'cancel' if user pressed ESC
            None otherwise
        """
        if key == 27:  # ESC
            return 'cancel'
        elif key == 13:  # Enter
            return 'confirm'
        elif key == 8:  # Backspace
            self.text = self.text[:-1]
        elif key == 32:  # Space
            self.text += " "
        elif 32 < key < 127:  # Printable chars
            self.text += chr(key)
        return None


# ==================== ON-SCREEN USER LIST ====================


def draw_user_list(display, users):
    """Draw a user list overlay on screen."""
    h, w = display.shape[:2]

    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)

    # Title
    title = f"Registered Users ({len(users)})"
    ts = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    cv2.putText(display, title, ((w - ts[0]) // 2, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_GREEN, 2)

    if not users:
        msg = "No users registered yet"
        ms = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(display, msg, ((w - ms[0]) // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GRAY, 2)
    else:
        # Draw user list
        start_y = 90
        for i, (user_id, name, _) in enumerate(users):
            if start_y + i * 40 > h - 80:
                cv2.putText(display, f"... and {len(users) - i} more", (80, start_y + i * 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1)
                break

            y_pos = start_y + i * 40
            # User row
            cv2.rectangle(display, (60, y_pos - 5), (w - 60, y_pos + 30), (50, 50, 50), -1)
            # ID badge
            cv2.circle(display, (85, y_pos + 12), 15, COLOR_BLUE, -1)
            cv2.putText(display, str(user_id), (78, y_pos + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
            # Name
            cv2.putText(display, name, (115, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)

    # Close instruction
    cv2.putText(display, "Press any key to close", ((w - 200) // 2, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1)

    return display


# ==================== DELETE USER SCREEN ====================


def delete_user_screen(cap, keyboard):
    """Handle user deletion with on-screen keyboard."""
    keyboard.reset("Enter name to delete")

    while True:
        ret, frame = cap.read()
        if not ret:
            return

        display = frame.copy()
        display = keyboard.draw(display)
        cv2.imshow("Face Scanner", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            continue

        result = keyboard.handle_key(key)
        if result == 'cancel':
            return
        elif result == 'confirm':
            name = keyboard.text.strip()
            if not name:
                continue

            deleted = delete_user(name)

            # Show result
            for _ in range(90):
                ret2, frame2 = cap.read()
                if not ret2:
                    break
                disp = frame2.copy()
                overlay = disp.copy()
                cv2.rectangle(overlay, (0, 0), (disp.shape[1], disp.shape[0]), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, disp, 0.3, 0, disp)

                if deleted:
                    msg = f"User '{name}' deleted successfully"
                    color = COLOR_GREEN
                else:
                    msg = f"User '{name}' not found"
                    color = COLOR_RED

                ms = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.putText(disp, msg, ((disp.shape[1] - ms[0]) // 2, disp.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.imshow("Face Scanner", disp)
                cv2.waitKey(33)
            return


# ==================== MAIN MODES ====================


def register_mode(cap, net, keyboard):
    """Registration mode - all in GUI."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        faces = detect_faces(frame, net)

        draw_top_bar(display, "REGISTER MODE", COLOR_GREEN)

        if len(faces) == 1:
            x, y, w, h = faces[0]
            draw_face_frame(display, x, y, w, h, COLOR_GREEN, 3)
            draw_center_message(display, "Face detected - Press SPACE to capture", COLOR_GREEN)
        elif len(faces) == 0:
            draw_center_message(display, "No face detected - Look at camera", COLOR_RED)
        else:
            for x, y, w, h in faces:
                draw_face_frame(display, x, y, w, h, COLOR_YELLOW, 2)
            draw_center_message(display, "Multiple faces - Only one person please", COLOR_YELLOW)

        draw_bottom_bar(display, "SPACE = Capture  |  ESC = Back to Menu")
        cv2.imshow("Face Scanner", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return
        elif key == 32:
            if len(faces) != 1:
                continue

            encoding = get_face_encoding(frame, faces[0])
            if encoding is None:
                continue

            # Check if already registered
            already, existing_name, score = is_face_already_registered(encoding)
            if already:
                for _ in range(90):
                    ret2, frame2 = cap.read()
                    if not ret2:
                        break
                    disp = frame2.copy()
                    draw_top_bar(disp, "ALREADY REGISTERED", COLOR_YELLOW)
                    draw_center_message(disp, f"Face belongs to '{existing_name}' ({score:.0f}%)", COLOR_YELLOW, -15)
                    draw_center_message(disp, "Cannot register same face twice", COLOR_RED, 25)
                    cv2.imshow("Face Scanner", disp)
                    cv2.waitKey(33)
                continue

            # Show on-screen keyboard for name input
            keyboard.reset("Enter Your Name")

            while True:
                ret2, frame2 = cap.read()
                if not ret2:
                    return

                disp = frame2.copy()
                disp = keyboard.draw(disp)
                cv2.imshow("Face Scanner", disp)

                k = cv2.waitKey(1) & 0xFF
                if k == 255:
                    continue

                result = keyboard.handle_key(k)
                if result == 'cancel':
                    break
                elif result == 'confirm':
                    name = keyboard.text.strip()
                    if not name:
                        continue

                    success = save_user(name, encoding)

                    # Show result
                    for _ in range(90):
                        ret3, frame3 = cap.read()
                        if not ret3:
                            break
                        disp3 = frame3.copy()
                        if success:
                            draw_top_bar(disp3, "REGISTERED!", COLOR_GREEN)
                            draw_center_message(disp3, f"'{name}' added successfully!", COLOR_GREEN, -15)
                            draw_center_message(disp3, f"Total users: {get_user_count()}", COLOR_WHITE, 25)
                        else:
                            draw_top_bar(disp3, "ERROR", COLOR_RED)
                            draw_center_message(disp3, f"Name '{name}' already taken!", COLOR_RED)
                        cv2.imshow("Face Scanner", disp3)
                        cv2.waitKey(33)
                    return
            # If cancelled from keyboard, continue registration loop


def verify_mode(cap, net):
    """Verification mode - all in GUI."""
    if get_user_count() == 0:
        for _ in range(90):
            ret, frame = cap.read()
            if not ret:
                break
            disp = frame.copy()
            draw_top_bar(disp, "NO USERS", COLOR_RED)
            draw_center_message(disp, "No users registered! Register first.", COLOR_RED)
            draw_bottom_bar(disp, "Press any key to go back")
            cv2.imshow("Face Scanner", disp)
            if cv2.waitKey(33) & 0xFF != 255:
                break
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        faces = detect_faces(frame, net)

        draw_top_bar(display, "VERIFY MODE", COLOR_BLUE)

        if len(faces) == 1:
            x, y, w, h = faces[0]
            draw_face_frame(display, x, y, w, h, COLOR_BLUE, 3)
            draw_center_message(display, "Face detected - Press SPACE to verify", COLOR_BLUE)
        elif len(faces) == 0:
            draw_center_message(display, "No face detected - Look at camera", COLOR_RED)
        else:
            for x, y, w, h in faces:
                draw_face_frame(display, x, y, w, h, COLOR_YELLOW, 2)
            draw_center_message(display, "Multiple faces - Only one person please", COLOR_YELLOW)

        draw_bottom_bar(display, "SPACE = Verify  |  ESC = Back to Menu")
        cv2.imshow("Face Scanner", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return
        elif key == 32:
            if len(faces) != 1:
                continue

            encoding = get_face_encoding(frame, faces[0])
            if encoding is None:
                continue

            # Scanning animation
            for i in range(40):
                ret2, frame2 = cap.read()
                if not ret2:
                    break
                disp = frame2.copy()
                draw_top_bar(disp, "SCANNING...", COLOR_YELLOW)
                scan_y = int(i / 40 * frame2.shape[0])
                cv2.line(disp, (0, scan_y), (frame2.shape[1], scan_y), COLOR_GREEN, 2)
                dots = "." * ((i // 8) % 4)
                draw_center_message(disp, f"Verifying identity{dots}", COLOR_YELLOW)
                cv2.imshow("Face Scanner", disp)
                cv2.waitKey(33)

            # Verify
            verified, name, confidence = verify_face(encoding)

            # Show result
            for _ in range(120):
                ret2, frame2 = cap.read()
                if not ret2:
                    break
                disp = frame2.copy()
                disp = draw_result_screen(disp, verified, name, confidence)
                cv2.imshow("Face Scanner", disp)
                cv2.waitKey(33)
            return


# ==================== MAIN ====================


def draw_menu(cap):
    """Draw the on-screen main menu."""
    ret, frame = cap.read()
    if not ret:
        return None

    h, w = frame.shape[:2]
    display = frame.copy()

    # Darken
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

    # Title
    title = "FACE SCANNER"
    ts = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)[0]
    cv2.putText(display, title, ((w - ts[0]) // 2, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.3, COLOR_GREEN, 3)

    subtitle = "Verification System"
    ss = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    cv2.putText(display, subtitle, ((w - ss[0]) // 2, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GRAY, 1)

    # Menu panel
    panel_x = w // 2 - 160
    panel_y = 110
    panel_w = 320
    panel_h = 280

    overlay2 = display.copy()
    cv2.rectangle(overlay2, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay2, 0.85, display, 0.15, 0, display)
    cv2.rectangle(display, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (80, 80, 80), 1)

    # Menu items
    items = [
        ("1", "Register Face", COLOR_GREEN),
        ("2", "Verify Face", COLOR_BLUE),
        ("3", "List Users", COLOR_YELLOW),
        ("4", "Delete User", COLOR_RED),
        ("5", "Quit", COLOR_GRAY),
    ]

    for i, (key, label, color) in enumerate(items):
        iy = panel_y + 45 + i * 45

        # Key badge
        cv2.rectangle(display, (panel_x + 25, iy - 15), (panel_x + 60, iy + 15), color, -1)
        kts = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(display, key, (panel_x + 25 + (35 - kts[0]) // 2, iy + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_BLACK, 2)

        # Label
        cv2.putText(display, label, (panel_x + 75, iy + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 1)

    # User count at bottom
    count = get_user_count()
    count_text = f"Registered Users: {count}"
    cv2.putText(display, count_text, (panel_x + 25, panel_y + panel_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1)

    # Bottom bar
    draw_bottom_bar(display, "Press 1-5 to select  |  All controls on-screen")

    return display


def main():
    """Main app loop - everything happens in the GUI window."""
    init_db()

    print("Face Scanner starting...")
    net = load_face_detector()
    print("Ready! All controls are in the app window.")
    print("(You can minimize this terminal)\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    keyboard = OnScreenKeyboard()

    try:
        while True:
            menu_frame = draw_menu(cap)
            if menu_frame is None:
                break

            cv2.imshow("Face Scanner", menu_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("1"):
                register_mode(cap, net, keyboard)
            elif key == ord("2"):
                verify_mode(cap, net)
            elif key == ord("3"):
                # Show user list on screen
                ret, frame = cap.read()
                if ret:
                    users = get_all_users()
                    disp = draw_user_list(frame, users)
                    cv2.imshow("Face Scanner", disp)
                    cv2.waitKey(0)  # Wait for any key
            elif key == ord("4"):
                delete_user_screen(cap, keyboard)
            elif key == ord("5") or key == ord("q") or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Face scanner closed.")


if __name__ == "__main__":
    main()
