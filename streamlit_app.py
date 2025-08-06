"""
Entry point for Streamlit Cloud deployment
This file allows the app to run directly from the root directory
"""

import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main app
from app import main

if __name__ == "__main__":
    main()
