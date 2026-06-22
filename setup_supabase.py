"""
Setup script to create the required table in Supabase.
Run this ONCE before using the app.
"""

import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")


def setup():
    print("Connecting to Supabase...")
    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Create the users table using SQL via Supabase's REST API
    # NOTE: You need to run this SQL in the Supabase Dashboard → SQL Editor
    sql = """
    CREATE TABLE IF NOT EXISTS face_users (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        face_encoding BYTEA NOT NULL,
        encoding_size INTEGER NOT NULL,
        registered_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_face_users_name ON face_users(name);
    """

    print("\n" + "=" * 50)
    print("  SUPABASE SETUP")
    print("=" * 50)
    print("\nPlease run this SQL in your Supabase Dashboard:")
    print("  1. Go to: https://supabase.com/dashboard")
    print("  2. Select your project")
    print("  3. Go to SQL Editor (left sidebar)")
    print("  4. Click 'New Query'")
    print("  5. Paste the following SQL and click 'Run':")
    print("\n" + "-" * 50)
    print(sql)
    print("-" * 50)

    # Test connection
    print("\nTesting connection...")
    try:
        result = client.table("face_users").select("*").limit(1).execute()
        print("✓ Connected successfully! Table 'face_users' exists.")
        print(f"  Current rows: {len(result.data)}")
    except Exception as e:
        if "relation" in str(e) and "does not exist" in str(e):
            print("✗ Table 'face_users' does not exist yet.")
            print("  Please run the SQL above in the Supabase Dashboard first.")
        else:
            print(f"✗ Connection error: {e}")


if __name__ == "__main__":
    setup()
