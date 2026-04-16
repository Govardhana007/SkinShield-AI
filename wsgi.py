"""
WSGI entrypoint for Render deployment
This imports the Flask app from skin_project/app.py and runs it
"""
import sys
import os

# Add skin_project to Python path FIRST (before any imports)
skin_project_path = os.path.join(os.path.dirname(__file__), 'skin_project')
sys.path.insert(0, skin_project_path)

# Change to skin_project directory for relative imports
os.chdir(skin_project_path)

# NOW import the Flask app from skin_project/app.py
from app import app

if __name__ == "__main__":
    # Get PORT from environment or default to 5000
    port = int(os.getenv("PORT", 5000))
    # Bind to 0.0.0.0 so Render can access it
    app.run(host="0.0.0.0", port=port, debug=False)
