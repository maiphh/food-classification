"""
Deployment preparation script
This script prepares your app for deployment to various hosting platforms
"""

import os
import shutil
import json

def create_deployment_files():
    """Create necessary files for deployment."""
    
    print("🚀 Preparing app for deployment...")
    
    # Create .gitignore if it doesn't exist
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Temporary files
temp/
tmp/
"""

    with open('.gitignore', 'w') as f:
        f.write(gitignore_content.strip())
    print("✅ Created .gitignore")
    
    # Update requirements.txt for deployment
    requirements_content = """streamlit>=1.28.0
tensorflow>=2.10.0
numpy>=1.21.0
pillow>=8.3.0
pandas>=1.3.0"""

    with open('requirements.txt', 'w') as f:
        f.write(requirements_content)
    print("✅ Updated requirements.txt for deployment")
    
    # Create a simple config.toml for Streamlit
    os.makedirs('.streamlit', exist_ok=True)
    config_content = """[theme]
primaryColor = "#FF6B35"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
maxUploadSize = 10
"""
    
    with open('.streamlit/config.toml', 'w') as f:
        f.write(config_content)
    print("✅ Created Streamlit configuration")
    
    # Create deployment README
    readme_content = """# 🍕 Food Classification App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

Upload food images and get AI-powered predictions for 101 different food categories!

## 🚀 Try it Live
**[Launch the App](https://your-app-name.streamlit.app)** *(Update this link after deployment)*

## 📱 Features
- 📸 Upload any food image
- 🤖 AI classification with confidence scores
- 📊 Top predictions visualization
- 🎨 Beautiful, responsive interface

## 🍽️ Supported Foods
Pizza, Sushi, Hamburger, Ice Cream, Chocolate Cake, and 96 more categories!

## 🔧 Local Development
```bash
pip install -r requirements.txt
streamlit run src/app.py
```

## 🤖 Model
Fine-tuned MobileNetV2 on Food-101 dataset
"""

    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("✅ Created deployment README.md")
    
    print("\n🎯 Deployment Ready!")
    print("=" * 40)
    print("Your app is now ready for deployment to:")
    print("1. 🌟 Streamlit Community Cloud (Recommended)")
    print("2. 🤗 Hugging Face Spaces")
    print("3. ⚡ Railway")
    print("4. 🚀 Render")
    print("\nSee DEPLOYMENT_GUIDE.md for detailed instructions!")

def check_deployment_readiness():
    """Check if the app is ready for deployment."""
    print("🔍 Checking deployment readiness...")
    
    required_files = [
        'src/app.py',
        'streamlit_app.py',
        'requirements.txt',
        'notebook/converted_model.keras',
        'data/meta/classes.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    # Check model file size
    model_path = 'notebook/converted_model.keras'
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"📊 Model size: {size_mb:.2f} MB")
        if size_mb > 100:
            print("⚠️  Model size exceeds GitHub's 100MB limit")
            print("   Consider using Git LFS or hosting model elsewhere")
    
    print("✅ All required files present!")
    return True

def main():
    """Main preparation function."""
    print("🛠️  Food Classification App - Deployment Preparation")
    print("=" * 55)
    
    # Check readiness
    if check_deployment_readiness():
        print()
        create_deployment_files()
        
        print("\n📋 Next Steps:")
        print("1. Push code to GitHub repository")
        print("2. Go to share.streamlit.io")
        print("3. Deploy your app")
        print("4. Share your live app URL!")
    else:
        print("\n❌ Please fix missing files before deployment")

if __name__ == "__main__":
    main()
