"""
Food Classification Streamlit App

A web application that allows users to upload food images and get predictions
from a trained MobileNetV2 model for 101 different food categories.
"""

import streamlit as st
import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd

# Add the src directory to Python path to import custom modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from util import get_data_list


def load_model_robust(model_path):
    """
    Robust model loading with multiple fallback approaches.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        tensorflow.keras.Model: Loaded model or None if failed
    """
    loading_strategies = [
        # Strategy 1: Standard loading
        lambda path: tf.keras.models.load_model(path),
        
        # Strategy 2: Load without compilation
        lambda path: tf.keras.models.load_model(path, compile=False),
        
        # Strategy 3: Load with safe_mode=False (for newer Keras versions)
        lambda path: tf.keras.models.load_model(path, compile=False, safe_mode=False),
        
        # Strategy 4: Load with custom objects handling
        lambda path: tf.keras.models.load_model(
            path, 
            compile=False,
            custom_objects={'Functional': tf.keras.Model}
        ),
    ]
    
    for i, strategy in enumerate(loading_strategies, 1):
        try:
            st.info(f"Trying loading strategy {i}...")
            model = strategy(model_path)
            st.success(f"Successfully loaded model using strategy {i}")
            return model
        except Exception as e:
            st.warning(f"Strategy {i} failed: {str(e)[:100]}...")
            continue
    
    return None


class FoodClassifierApp:
    def __init__(self):
        """Initialize the Food Classifier App."""
        self.model = None
        self.class_names = []
        
    @st.cache_resource
    def load_model_and_classes(_self):
        """Load the trained model and class names. Cached for performance."""
        try:
            # Get model path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            
            # Debug info for deployment
            st.write(f"🔍 Debug info:")
            st.write(f"Script dir: {script_dir}")
            st.write(f"Project root: {project_root}")
            
            # Check what's available
            if os.path.exists(os.path.join(project_root, 'notebook')):
                notebook_files = os.listdir(os.path.join(project_root, 'notebook'))
                st.write(f"Notebook files: {notebook_files}")
            else:
                st.error("Notebook directory not found!")
            
            # Try different model files in order of preference
            model_files = [
                'converted_model.keras',      # Our converted model (highest priority)
                'food101_mobilenetv2_finetuned.keras',
                'food101_mobilenetv2.keras',
                'best_model.h5'
            ]
            
            model = None
            model_path = None
            
            for model_file in model_files:
                temp_path = os.path.join(project_root, 'notebook', model_file)
                if os.path.exists(temp_path):
                    try:

                        
                        # Use robust loading function
                        model = load_model_robust(temp_path)
                        
                        if model is not None:
                            model_path = temp_path
                            break
                        
                    except Exception as load_error:
                        continue
            
            if model is None:
                return None, []
            
            # Load class names
            classes_path = os.path.join(project_root, 'data', 'meta', 'classes.txt')
            if not os.path.exists(classes_path):
                # Try alternative path for cloud deployment (data folder might be ignored)
                classes_path = os.path.join(script_dir, 'classes.txt')
                if not os.path.exists(classes_path):
                    st.error(f"Classes file not found in any location")
                    return None, []
                
            class_names = get_data_list(classes_path)
            
            return model, class_names
            
        except Exception as e:
            st.error(f"Error loading model or classes: {e}")
            st.error("Please check that the model files exist in the 'notebook' directory")
            return None, []
    
    def preprocess_image(self, image):
        """
        Preprocess an image for prediction.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            numpy.ndarray: Preprocessed image array
        """
        try:
            # Resize image to 224x224 (MobileNetV2 input size)
            img = image.resize((224, 224))
            
            # Convert to array and normalize
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize to [0, 1]
            
            return img_array
        except Exception as e:
            st.error(f"Error preprocessing image: {e}")
            return None
    
    def predict(self, image, top_k=5):
        """
        Predict the class of an image.
        
        Args:
            image (PIL.Image): Input image
            top_k (int): Number of top predictions to return
            
        Returns:
            list: List of tuples (class_name, confidence) for top predictions
        """
        # Load model and classes if not already loaded
        if self.model is None:
            self.model, self.class_names = self.load_model_and_classes()
        
        if self.model is None:
            st.error("Model failed to load. Cannot make predictions.")
            return None
        
        if not self.class_names:
            st.error("Class names failed to load. Cannot make predictions.")
            return None
        
        # Preprocess the image
        img_array = self.preprocess_image(image)
        if img_array is None:
            return None
        
        try:
            # Make prediction
            with st.spinner('Making prediction...'):
                predictions = self.model.predict(img_array, verbose=0)
                predictions = predictions[0]  # Remove batch dimension
            
            # Get top k predictions
            top_indices = np.argsort(predictions)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                class_name = self.class_names[idx]
                confidence = float(predictions[idx])
                results.append((class_name, confidence))
            
            return results
        
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            return None


def main():
    """Main Streamlit application."""
    # Set page config
    st.set_page_config(
        page_title="Food Classification App",
        page_icon="🍕",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("🍕 Food Classification App")
    st.markdown("""
    Upload an image of food and get AI-powered predictions for what type of food it is!
    This app can classify **101 different types of food** using a fine-tuned MobileNetV2 model.
    """)
    
    # Initialize the classifier
    classifier = FoodClassifierApp()
    
    # Sidebar
    st.sidebar.title("📋 Instructions")
    st.sidebar.markdown("""
    1. Upload a food image using the file uploader
    2. The app will automatically predict the food type
    3. View the top predictions with confidence scores
    4. Try different food images to test the model!
    """)
    
    st.sidebar.title("🍽️ Supported Food Types")
    st.sidebar.markdown("""
    The model can classify 101 different food categories including:
    - **Italian**: Pizza, Lasagna, Spaghetti, Ravioli
    - **Asian**: Sushi, Ramen, Pad Thai, Dumplings
    - **American**: Hamburger, Hot Dog, French Fries
    - **Desserts**: Ice Cream, Chocolate Cake, Donuts
    - **And many more!**
    """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Food Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a food image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of food for best results"
        )
        
        # Top-k selector
        top_k = st.slider(
            "Number of predictions to show:",
            min_value=1,
            max_value=10,
            value=5,
            help="Select how many top predictions to display"
        )
    
    with col2:
        st.subheader("🔍 Predictions")
        
        if uploaded_file is not None:
            try:
                # Display the uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Make prediction
                results = classifier.predict(image, top_k=top_k)
                
                if results:
                    # Display results
                    st.subheader("🎯 Top Predictions:")
                    
                    # Create a DataFrame for better display
                    df_results = pd.DataFrame(results, columns=['Food Type', 'Confidence'])
                    df_results['Food Type'] = df_results['Food Type'].str.replace('_', ' ').str.title()
                    df_results['Confidence'] = (df_results['Confidence'] * 100).round(2)
                    df_results['Confidence'] = df_results['Confidence'].astype(str) + '%'
                    df_results.index = range(1, len(df_results) + 1)
                    
                    # Display as a styled table
                    st.dataframe(
                        df_results,
                        use_container_width=True,
                        hide_index=False
                    )
                    
                    # Highlight top prediction
                    top_food = results[0][0].replace('_', ' ').title()
                    top_confidence = results[0][1] * 100
                    
                    if top_confidence > 50:
                        st.success(f"🎉 **Prediction: {top_food}** (Confidence: {top_confidence:.2f}%)")
                    elif top_confidence > 25:
                        st.warning(f"🤔 **Likely: {top_food}** (Confidence: {top_confidence:.2f}%)")
                    else:
                        st.info(f"🤷 **Possible: {top_food}** (Confidence: {top_confidence:.2f}%)")
                    
                    # Progress bars for top 3 predictions
                    st.subheader("📊 Confidence Visualization")
                    for i, (food_name, confidence) in enumerate(results[:3]):
                        formatted_name = food_name.replace('_', ' ').title()
                        st.write(f"**{i+1}. {formatted_name}**")
                        st.progress(confidence)
                        st.write(f"{confidence*100:.2f}%")
                        st.write("")
                
                else:
                    st.error("Failed to make prediction. Please try again with a different image.")
                    
            except Exception as e:
                st.error(f"Error processing image: {e}")
        
        else:
            st.info("👆 Please upload a food image to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🤖 Powered by TensorFlow and MobileNetV2 | Built with Streamlit</p>
        <p>📊 Trained on the Food-101 dataset with 101 food categories</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
