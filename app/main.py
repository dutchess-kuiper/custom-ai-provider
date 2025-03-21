"""
Module that imports and re-exports the FastAPI app from the root main.py file.
This allows Railway to find the app using the `app.main:app` import path.
"""

# Import the app from the root main.py
from main import app

# The app variable is now available for import as `from app.main import app` 