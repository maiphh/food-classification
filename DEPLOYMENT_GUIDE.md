# Deployment Guide - Free Hosting Options

## 🌟 Option 1: Streamlit Community Cloud (Recommended)

### Prerequisites
- GitHub account
- Your code in a GitHub repository

### Step-by-Step Deployment

1. **Prepare your repository structure**:
   ```
   your-repo/
   ├── src/
   │   └── app.py              # Main Streamlit app
   ├── notebook/
   │   └── converted_model.keras # Your model file
   ├── data/
   │   └── meta/
   │       └── classes.txt     # Class names
   ├── requirements.txt        # Dependencies
   └── README.md              # Project description
   ```

2. **Create/Update requirements.txt**:
   ```
   streamlit>=1.28.0
   tensorflow>=2.10.0
   numpy>=1.21.0
   pillow>=8.3.0
   pandas>=1.3.0
   ```

3. **Create a main app file** (if needed):
   Create `streamlit_app.py` in the root directory:
   ```python
   import sys
   import os
   sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
   
   from app import main
   
   if __name__ == "__main__":
       main()
   ```

4. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub account
   - Select your repository
   - Set the main file path: `src/app.py` or `streamlit_app.py`
   - Click "Deploy"

5. **Your app will be live at**: `https://your-repo-name.streamlit.app`

### Benefits
- ✅ 100% Free
- ✅ Automatic updates from GitHub
- ✅ Custom domain
- ✅ SSL certificate included
- ✅ Perfect for Streamlit apps

---

## 🤗 Option 2: Hugging Face Spaces

### Steps
1. **Create account** at [huggingface.co](https://huggingface.co)
2. **Create new Space**:
   - Click "Create new Space"
   - Choose "Streamlit" as SDK
   - Set repository name
3. **Upload your files**:
   - Upload all your code files
   - Ensure `requirements.txt` is included
   - Main app should be named `app.py`
4. **Automatic deployment** happens after upload

### Benefits
- ✅ Free hosting
- ✅ Great for ML models
- ✅ Built-in model hosting
- ✅ Community features

---

## ⚙️ Option 3: Railway

### Steps
1. **Sign up** at [railway.app](https://railway.app)
2. **Connect GitHub** repository
3. **Configure build**:
   - Runtime: Python
   - Start command: `streamlit run src/app.py --server.port $PORT`
4. **Deploy**

### Benefits
- ✅ Free tier available
- ✅ Automatic deployments
- ✅ Custom domains
- ✅ Database support

---

## 🚀 Option 4: Render

### Steps
1. **Sign up** at [render.com](https://render.com)
2. **Create Web Service** from GitHub repo
3. **Configure**:
   - Runtime: Python 3
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run src/app.py --server.port $PORT --server.address 0.0.0.0`

### Benefits
- ✅ Free tier with good limits
- ✅ Automatic SSL
- ✅ Custom domains
- ✅ Good performance

---

## 📋 Pre-Deployment Checklist

### File Structure ✅
- [ ] All code files in repository
- [ ] Model file (`converted_model.keras`) included
- [ ] Class names file (`classes.txt`) included
- [ ] Proper `requirements.txt`

### Code Adjustments ✅
- [ ] Relative paths work correctly
- [ ] No hardcoded absolute paths
- [ ] Environment variables for sensitive data
- [ ] Proper error handling for missing files

### Testing ✅
- [ ] App runs locally: `streamlit run src/app.py`
- [ ] All dependencies listed in requirements.txt
- [ ] Model loads successfully
- [ ] File paths work in different environments

---

## 🛠️ Troubleshooting Common Issues

### Model File Too Large
- **Problem**: GitHub has 100MB file limit
- **Solution**: Use Git LFS or host model elsewhere
- **Alternative**: Use Hugging Face model repository

### Memory Issues
- **Problem**: Free tiers have memory limits
- **Solution**: Optimize model loading with `@st.cache_resource`
- **Alternative**: Use model quantization

### Slow Loading
- **Problem**: Model takes time to load
- **Solution**: Use Streamlit's caching decorators
- **Alternative**: Lazy loading techniques

---

## 🎯 Recommended Deployment Flow

1. **Start with Streamlit Community Cloud** (easiest)
2. **If model too large**: Try Hugging Face Spaces
3. **If need more control**: Use Railway or Render
4. **For production**: Consider paid hosting options

**Your food classification app will be live and accessible to everyone!** 🌐
