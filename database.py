"""
Database module using MySQL.
Handles face encoding storage, retrieval, and attendance logging.
"""

import os
import base64
import datetime
import numpy as np
from dotenv import load_dotenv
import mysql.connector

load_dotenv()

# Database config
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Algo@123")
DB_DATABASE = os.getenv("DB_DATABASE", "jwellery")

def get_connection():
    """Get a connection to the MySQL database."""
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_DATABASE
    )

def init_db():
    """Initialize face scanner and attendance tables if they do not exist."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Create employee_faces table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS employee_faces (
            employee_id VARCHAR(50) PRIMARY KEY,
            face_encoding TEXT NOT NULL,
            encoding_size INT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE
        ) ENGINE=InnoDB;
        """)
        
        # Create attendance table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INT AUTO_INCREMENT PRIMARY KEY,
            employee_id VARCHAR(50) NOT NULL,
            date DATE NOT NULL,
            check_in TIME DEFAULT NULL,
            check_out TIME DEFAULT NULL,
            working_hours DECIMAL(5,2) DEFAULT 0.00,
            overtime_hours DECIMAL(5,2) DEFAULT 0.00,
            status VARCHAR(20) DEFAULT 'present',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY emp_date (employee_id, date),
            FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE
        ) ENGINE=InnoDB;
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        print("  Database tables verified/created successfully.")
    except Exception as e:
        print(f"  ERROR: MySQL database connection or table creation failed: {e}")

def save_user(employee_id, face_encoding):
    """
    Save a new face template linked to an employee.
    """
    conn = get_connection()
    cursor = conn.cursor()
    encoding_b64 = base64.b64encode(face_encoding.tobytes()).decode("utf-8")
    try:
        cursor.execute(
            "INSERT INTO employee_faces (employee_id, face_encoding, encoding_size) VALUES (%s, %s, %s) "
            "ON DUPLICATE KEY UPDATE face_encoding = VALUES(face_encoding), encoding_size = VALUES(encoding_size)",
            (employee_id, encoding_b64, int(face_encoding.shape[0]))
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"  DB Error saving user: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def get_all_users():
    """
    Retrieve all registered users and their face encodings.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # Join with employees table to get the full name
        cursor.execute(
            "SELECT ef.employee_id, CONCAT(e.first_name, ' ', e.last_name) AS name, ef.face_encoding "
            "FROM employee_faces ef "
            "JOIN employees e ON ef.employee_id = e.employee_id"
        )
        rows = cursor.fetchall()
    except Exception as e:
        print(f"  DB Error fetching users: {e}")
        rows = []
    finally:
        cursor.close()
        conn.close()

    users = []
    for emp_id, name, encoding_b64 in rows:
        encoding_bytes = base64.b64decode(encoding_b64)
        face_encoding = np.frombuffer(encoding_bytes, dtype=np.float32).reshape(-1)
        users.append((emp_id, name, face_encoding))

    return users

def delete_user(employee_id):
    """Delete employee face data by employee_id."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM employee_faces WHERE employee_id = %s", (employee_id,))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"  DB Error deleting user: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def get_user_count():
    """Get total number of enrolled faces."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM employee_faces")
        count = cursor.fetchone()[0]
    except Exception as e:
        print(f"  DB Error counting users: {e}")
        count = 0
    finally:
        cursor.close()
        conn.close()
    return count

def mark_attendance(employee_id):
    """
    Mark check-in or check-out for today, and calculate worked and overtime hours.
    """
    conn = get_connection()
    cursor = conn.cursor()
    today = datetime.date.today().isoformat()
    now_time = datetime.datetime.now().time().strftime("%H:%M:%S")

    try:
        # Check if record exists for today
        cursor.execute("SELECT id, check_in, check_out FROM attendance WHERE employee_id = %s AND date = %s", (employee_id, today))
        row = cursor.fetchone()

        if not row:
            # Check-in flow
            cursor.execute(
                "INSERT INTO attendance (employee_id, date, check_in, status) VALUES (%s, %s, %s, 'present')",
                (employee_id, today, now_time)
            )
            conn.commit()
            return {
                "success": True,
                "action": "check_in",
                "time": now_time,
                "message": f"Checked In successfully at {now_time}"
            }
        else:
            # Check-out flow or already checked out
            att_id, check_in_time, check_out_time = row
            
            def format_td(td):
                if isinstance(td, datetime.timedelta):
                    hours, remainder = divmod(td.seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                return str(td)

            if check_out_time is None:
                return {
                    "success": True,
                    "action": "needs_checkout",
                    "time": now_time,
                    "check_in": format_td(check_in_time),
                    "message": "Already checked in today. Please click checkout."
                }
            else:
                return {
                    "success": True,
                    "action": "already_checked_out",
                    "time": now_time,
                    "check_in": format_td(check_in_time),
                    "check_out": format_td(check_out_time),
                    "message": "Today shift off. Checkin markup tomorrow."
                }

    except Exception as e:
        print(f"  DB Error marking attendance: {e}")
        return {"success": False, "message": f"Database error: {str(e)}"}
    finally:
        cursor.close()
        conn.close()

def manual_checkout(employee_id):
    """
    Mark check-out for today manually, and calculate worked and overtime hours.
    """
    conn = get_connection()
    cursor = conn.cursor()
    today = datetime.date.today().isoformat()
    now_time = datetime.datetime.now().time().strftime("%H:%M:%S")

    try:
        cursor.execute("SELECT id, check_in, check_out FROM attendance WHERE employee_id = %s AND date = %s", (employee_id, today))
        row = cursor.fetchone()

        if not row:
            return {"success": False, "message": "No check-in record found for today."}

        att_id, check_in_time, check_out_time = row

        if check_out_time is not None:
            return {"success": False, "message": "Already checked out today."}

        # Update check_out and calculate working hours
        cursor.execute("""
            UPDATE attendance SET 
                check_out = %s,
                working_hours = ROUND(TIME_TO_SEC(TIMEDIFF(%s, check_in)) / 3600.0, 2),
                overtime_hours = ROUND(GREATEST(0.00, (TIME_TO_SEC(TIMEDIFF(%s, check_in)) / 3600.0) - 8.00), 2)
            WHERE id = %s
        """, (now_time, now_time, now_time, att_id))
        conn.commit()

        # Retrieve updated stats
        cursor.execute("SELECT check_in, check_out, working_hours, overtime_hours FROM attendance WHERE id = %s", (att_id,))
        updated_row = cursor.fetchone()

        def format_td(td):
            if isinstance(td, datetime.timedelta):
                hours, remainder = divmod(td.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            return str(td)

        ch_in = format_td(updated_row[0])
        ch_out = format_td(updated_row[1])
        w_hours = float(updated_row[2])
        o_hours = float(updated_row[3])

        return {
            "success": True,
            "action": "check_out",
            "time": now_time,
            "check_in": ch_in,
            "check_out": ch_out,
            "working_hours": w_hours,
            "overtime_hours": o_hours,
            "message": f"Checked Out successfully at {ch_out}. Total hours: {w_hours} (Overtime: {o_hours})"
        }

    except Exception as e:
        print(f"  DB Error marking manual checkout: {e}")
        return {"success": False, "message": f"Database error: {str(e)}"}
    finally:
        cursor.close()
        conn.close()
