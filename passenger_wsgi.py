import sys, os

# Add current directory to the Python path
sys.path.append(os.getcwd())

# Import the Flask app object from app.py and name it 'application'
from app import app as application
