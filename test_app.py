"""
Test script to verify model loading and basic functionality
"""

import os
import sys
import tensorflow as tf
from PIL import Image
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_model_loading():
    """Test if we can load the model successfully."""
    print("🔍 Testing model loading...")
    
    # Get model paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_dir = os.path.join(script_dir, 'notebook')
    
    # Check available model files
    model_files = [f for f in os.listdir(notebook_dir) if f.endswith(('.keras', '.h5'))]
    print(f"📁 Found model files: {model_files}")
    
    # Try loading each model
    for model_file in model_files:
        model_path = os.path.join(notebook_dir, model_file)
        print(f"\n🤖 Testing: {model_file}")
        
        try:
            # Try different loading approaches
            strategies = [
                ("Standard loading", lambda: tf.keras.models.load_model(model_path)),
                ("No compilation", lambda: tf.keras.models.load_model(model_path, compile=False)),
                ("Safe mode off", lambda: tf.keras.models.load_model(model_path, compile=False, safe_mode=False)),
            ]
            
            model = None
            for strategy_name, strategy_func in strategies:
                try:
                    print(f"   Trying: {strategy_name}")
                    model = strategy_func()
                    print(f"   ✅ Success with: {strategy_name}")
                    break
                except Exception as e:
                    print(f"   ❌ Failed: {str(e)[:100]}...")
                    continue
            
            if model is not None:
                print(f"   📊 Input shape: {model.input_shape}")
                print(f"   📊 Output shape: {model.output_shape}")
                
                # Test prediction with dummy data
                dummy_input = np.random.random((1, 224, 224, 3))
                try:
                    prediction = model.predict(dummy_input, verbose=0)
                    print(f"   🎯 Prediction test: Success (shape: {prediction.shape})")
                except Exception as e:
                    print(f"   🎯 Prediction test: Failed - {e}")
                
                return model, model_file
            else:
                print(f"   ❌ All strategies failed for {model_file}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return None, None

def test_classes_loading():
    """Test if we can load the class names."""
    print("\n🏷️ Testing class names loading...")
    
    try:
        from util import get_data_list
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        classes_path = os.path.join(script_dir, 'data', 'meta', 'classes.txt')
        
        if os.path.exists(classes_path):
            class_names = get_data_list(classes_path)
            print(f"   ✅ Loaded {len(class_names)} classes")
            print(f"   📝 First 5 classes: {class_names[:5]}")
            return class_names
        else:
            print(f"   ❌ Classes file not found: {classes_path}")
            return None
            
    except Exception as e:
        print(f"   ❌ Error loading classes: {e}")
        return None

def test_image_processing():
    """Test image preprocessing."""
    print("\n🖼️ Testing image processing...")
    
    try:
        # Create a dummy image
        dummy_image = Image.new('RGB', (300, 300), color='red')
        
        # Resize and process
        img_resized = dummy_image.resize((224, 224))
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        print(f"   ✅ Image processing successful")
        print(f"   📊 Processed shape: {img_array.shape}")
        print(f"   📊 Value range: {img_array.min():.3f} - {img_array.max():.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in image processing: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Food Classification App - Diagnostic Tests")
    print("=" * 50)
    
    # Test model loading
    model, model_file = test_model_loading()
    
    # Test class loading
    classes = test_classes_loading()
    
    # Test image processing
    image_ok = test_image_processing()
    
    # Summary
    print("\n📋 Test Summary:")
    print("=" * 30)
    print(f"Model loading: {'✅ PASS' if model is not None else '❌ FAIL'}")
    if model is not None:
        print(f"  Using: {model_file}")
    print(f"Classes loading: {'✅ PASS' if classes is not None else '❌ FAIL'}")
    if classes is not None:
        print(f"  Count: {len(classes)}")
    print(f"Image processing: {'✅ PASS' if image_ok else '❌ FAIL'}")
    
    if model is not None and classes is not None and image_ok:
        print("\n🎉 All tests passed! The app should work correctly.")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
