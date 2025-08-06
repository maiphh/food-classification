"""
Launch script for the Food Classification Streamlit App
"""
import subprocess
import sys
import os

def main():
    """Launch the Streamlit app."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, 'src', 'app.py')
    
    # Launch Streamlit
    print("🚀 Launching Food Classification App...")
    print("📱 The app will open in your default web browser")
    print("🛑 Press Ctrl+C to stop the app")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', app_path], check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching app: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
