"""
Database module using HTTP API Bridge.
Handles face encoding storage, retrieval, and attendance logging via PHP REST API.
"""

import base64
import numpy as np
import requests

# Bridge config
PHP_BRIDGE_URL = "https://test.algo2botsinfotech.com/Backend/face_db_bridge.php"
API_KEY = "slgold_face_super_secret_key_2026"

def _send_request(action, payload=None):
    if payload is None:
        payload = {}
    payload["api_key"] = API_KEY
    payload["action"] = action
    try:
        response = requests.post(PHP_BRIDGE_URL, json=payload, timeout=10)
        return response.json()
    except Exception as e:
        print(f"Bridge error ({action}): {e}")
        return {"success": False, "message": str(e)}

def init_db():
    res = _send_request("init_db")
    if res.get("success"):
        print("  Database tables verified/created successfully via bridge.")
    else:
        print(f"  ERROR: init_db failed via bridge: {res}")

def save_user(employee_id, face_encoding):
    encoding_b64 = base64.b64encode(face_encoding.tobytes()).decode("utf-8")
    payload = {
        "employee_id": employee_id,
        "face_encoding": encoding_b64,
        "encoding_size": int(face_encoding.shape[0])
    }
    res = _send_request("save_user", payload)
    return res.get("success", False)

def get_all_users():
    res = _send_request("get_all_users")
    users = []
    if res.get("success") and "users" in res:
        for u in res["users"]:
            emp_id = u["employee_id"]
            name = u["name"]
            encoding_b64 = u["face_encoding"]
            encoding_bytes = base64.b64decode(encoding_b64)
            face_encoding = np.frombuffer(encoding_bytes, dtype=np.float32).reshape(-1)
            users.append((emp_id, name, face_encoding))
    return users

def delete_user(employee_id):
    res = _send_request("delete_user", {"employee_id": employee_id})
    return res.get("deleted", False)

def get_user_count():
    res = _send_request("get_user_count")
    return res.get("count", 0)

def mark_attendance(employee_id):
    return _send_request("mark_attendance", {"employee_id": employee_id})

def manual_checkout(employee_id):
    return _send_request("manual_checkout", {"employee_id": employee_id})
