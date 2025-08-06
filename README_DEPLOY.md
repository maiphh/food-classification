# Food Classification App

A Streamlit web application for classifying food images using a fine-tuned MobileNetV2 model.

🌐 **[Live Demo on Streamlit Cloud](https://your-app-name.streamlit.app)** *(Deploy to get this link)*

## Features

- 📸 Upload food images (JPG, JPEG, PNG)
- 🤖 AI-powered classification of 101 food categories
- 📊 Top predictions with confidence scores
- 🎨 Beautiful, responsive web interface
- ⚡ Real-time predictions

## How to Use

1. Visit the live demo link above
2. Upload a food image using the file uploader
3. View predictions with confidence scores
4. Try different images to test accuracy!

## Supported Food Categories

The model can classify 101 different food types including:
- **Italian**: Pizza, Lasagna, Spaghetti, Ravioli
- **Asian**: Sushi, Ramen, Pad Thai, Dumplings  
- **American**: Hamburger, Hot Dog, French Fries
- **Desserts**: Ice Cream, Chocolate Cake, Donuts
- And many more!

## Technical Details

- **Model**: Fine-tuned MobileNetV2
- **Input**: 224x224 RGB images
- **Classes**: 101 food categories
- **Framework**: TensorFlow/Keras + Streamlit

## Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/food-classification-app.git
cd food-classification-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run src/app.py
```

## Model Information

The model was trained on the Food-101 dataset and achieves high accuracy on food image classification tasks. The converted model (`converted_model.keras`) is compatible with modern TensorFlow versions.

## License

MIT License - Feel free to use and modify!

---

**Built with ❤️ using Streamlit**
