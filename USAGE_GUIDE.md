# 🍕 Food Classification Streamlit App - Complete Guide

## ✅ Problem Solved!

The original model compatibility issue has been resolved by converting the model to a format compatible with your current TensorFlow version.

## 🚀 Quick Start

1. **Launch the app**:
   ```bash
   python run_app.py
   ```

2. **Open your browser** to `http://localhost:8501`

3. **Upload a food image** and get instant predictions!

## 📁 Files Created

### Core Application
- **`src/app.py`** - Main Streamlit application with web interface
- **`run_app.py`** - Simple launcher script
- **`requirements.txt`** - Updated with Streamlit dependency

### Utilities & Tools
- **`convert_model.py`** - Model conversion utility (fixes compatibility)
- **`test_app.py`** - Diagnostic test suite
- **`demo.py`** - Programmatic usage examples
- **`batch_classify.py`** - Batch processing for multiple images

### Model
- **`notebook/converted_model.keras`** - Compatible converted model (✅ Working)

## 🎯 Features

### Web Interface
- **Drag & drop image upload** (JPG, JPEG, PNG)
- **Real-time predictions** with confidence scores
- **Top-K predictions** (adjustable 1-10)
- **Visual confidence bars** for top 3 predictions
- **Color-coded results** based on confidence levels
- **Responsive design** with sidebar instructions

### Prediction Quality
- **High accuracy** on food images
- **101 food categories** supported
- **Confidence indicators**:
  - 🎉 Green: High confidence (>50%)
  - 🤔 Yellow: Medium confidence (25-50%)
  - 🤷 Blue: Low confidence (<25%)

## 📊 Model Information

- **Architecture**: MobileNetV2 with custom classification head
- **Input**: 224x224 RGB images
- **Output**: 101 food categories with confidence scores
- **Preprocessing**: Automatic resize and normalization
- **Source**: Converted from `best_model.h5` with preserved weights

## 🛠️ Troubleshooting

### If the app won't start:
```bash
# Install dependencies
pip install -r requirements.txt

# Test the setup
python test_app.py
```

### If predictions are poor:
- Use clear, well-lit food images
- Ensure food is the main subject
- Try different angles/lighting

### If model loading fails:
```bash
# Reconvert the model
python convert_model.py
```

## 💡 Usage Examples

### Web Interface
1. Go to `http://localhost:8501`
2. Upload an image using the file uploader
3. Adjust the number of predictions with the slider
4. View results in the right panel

### Programmatic Usage
```python
from src.app import FoodClassifierApp
from PIL import Image

# Initialize classifier
classifier = FoodClassifierApp()

# Load image
image = Image.open('path/to/food/image.jpg')

# Get predictions
results = classifier.predict(image, top_k=5)

# Display results
for i, (food_class, confidence) in enumerate(results, 1):
    print(f"{i}. {food_class.replace('_', ' ').title()}: {confidence*100:.2f}%")
```

### Batch Processing
```bash
# Process all images in a directory
python batch_classify.py path/to/images --output results.csv --top-k 3
```

## 🎉 Success!

Your food classification Streamlit app is now fully functional with:
- ✅ Working model loading
- ✅ Beautiful web interface
- ✅ Real-time predictions
- ✅ Multiple usage options
- ✅ Comprehensive error handling

**The app is ready to classify your food images!** 🍽️

---

## 📞 Support

If you encounter any issues:
1. Run `python test_app.py` to diagnose problems
2. Check that all files are in the correct locations
3. Ensure all dependencies are installed
4. Try the model conversion script if needed
