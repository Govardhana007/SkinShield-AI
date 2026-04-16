"""
Root entrypoint for Render deployment
This imports the Flask app from skin_project/app.py and runs it
"""
from app import app
import sys
import os

# Add skin_project to Python path so we can import its modules
skin_project_path = os.path.join(os.path.dirname(__file__), 'skin_project')
sys.path.insert(0, skin_project_path)
# Change to skin_project directory for relative imports
os.chdir(skin_project_path)

# Import the Flask app from skin_project/app.py

if __name__ == "__main__":
    app.run()
