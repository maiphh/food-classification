"""
Model conversion script to fix compatibility issues
This script will attempt to rebuild the model and convert it to a compatible format
"""

import os
import sys
import tensorflow as tf
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model import build_food101_model
from util import get_data_list

def convert_model():
    """Convert the model to a compatible format."""
    print("🔧 Model Conversion Utility")
    print("=" * 40)
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_dir = os.path.join(script_dir, 'notebook')
    
    # Load class names to get number of classes
    classes_path = os.path.join(script_dir, 'data', 'meta', 'classes.txt')
    class_names = get_data_list(classes_path)
    num_classes = len(class_names)
    
    print(f"📊 Number of classes: {num_classes}")
    
    # Try to load existing model weights
    model_files = [
        'food101_mobilenetv2_finetuned.keras',
        'food101_mobilenetv2.keras', 
        'best_model.h5'
    ]
    
    weights_loaded = False
    source_model_path = None
    
    for model_file in model_files:
        model_path = os.path.join(notebook_dir, model_file)
        if os.path.exists(model_path):
            print(f"\n🔍 Attempting to extract weights from: {model_file}")
            
            try:
                # Try different approaches to load weights only
                if model_file.endswith('.h5'):
                    # For .h5 files, try to load just the weights
                    temp_model, _ = build_food101_model(num_classes)
                    temp_model.load_weights(model_path)
                    weights_loaded = True
                    source_model_path = model_path
                    print(f"✅ Successfully loaded weights from {model_file}")
                    break
                else:
                    # For .keras files, try various loading strategies
                    strategies = [
                        lambda: tf.keras.models.load_model(model_path, compile=False),
                        lambda: tf.keras.models.load_model(model_path, compile=False, safe_mode=False),
                    ]
                    
                    for i, strategy in enumerate(strategies):
                        try:
                            old_model = strategy()
                            
                            # Create new model with same architecture
                            new_model, _ = build_food101_model(num_classes)
                            
                            # Try to copy weights
                            try:
                                new_model.set_weights(old_model.get_weights())
                                weights_loaded = True
                                source_model_path = model_path
                                temp_model = new_model
                                print(f"✅ Successfully loaded and converted weights from {model_file}")
                                break
                            except Exception as weight_error:
                                print(f"   ❌ Weight copying failed: {weight_error}")
                                continue
                                
                        except Exception as load_error:
                            print(f"   ❌ Strategy {i+1} failed: {str(load_error)[:100]}...")
                            continue
                    
                    if weights_loaded:
                        break
                        
            except Exception as e:
                print(f"   ❌ Failed to load {model_file}: {str(e)[:100]}...")
                continue
    
    if not weights_loaded:
        print("\n⚠️ Could not load existing model weights. Creating new untrained model...")
        temp_model, _ = build_food101_model(num_classes)
        print("📝 Note: This will be an untrained model for demonstration purposes.")
    
    # Save the converted model
    output_path = os.path.join(notebook_dir, 'converted_model.keras')
    
    try:
        print(f"\n💾 Saving converted model to: {output_path}")
        temp_model.save(output_path, save_format='keras')
        print("✅ Model saved successfully!")
        
        # Test loading the converted model
        print("\n🧪 Testing converted model...")
        test_model = tf.keras.models.load_model(output_path, compile=False)
        print(f"   📊 Input shape: {test_model.input_shape}")
        print(f"   📊 Output shape: {test_model.output_shape}")
        
        # Test prediction
        dummy_input = np.random.random((1, 224, 224, 3))
        prediction = test_model.predict(dummy_input, verbose=0)
        print(f"   🎯 Prediction test: Success (shape: {prediction.shape})")
        
        print("\n🎉 Conversion completed successfully!")
        print(f"   Source: {source_model_path if weights_loaded else 'New model'}")
        print(f"   Output: {output_path}")
        print(f"   Weights preserved: {'Yes' if weights_loaded else 'No (untrained)'}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to save converted model: {e}")
        return False

def main():
    """Main conversion function."""
    success = convert_model()
    
    if success:
        print("\n💡 Next steps:")
        print("   1. Update your app to use 'converted_model.keras'")
        print("   2. Test the Streamlit app: python run_app.py")
    else:
        print("\n❌ Conversion failed. Please check the errors above.")

if __name__ == "__main__":
    main()
