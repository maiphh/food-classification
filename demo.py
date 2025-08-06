"""
Demo script showing how to use the Food Classification app components
"""

import os
import sys
from PIL import Image

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from app import FoodClassifierApp

def demo_classification():
    """Demo the classification functionality with sample images."""
    print("🍕 Food Classification Demo")
    print("=" * 40)
    
    # Initialize the classifier
    classifier = FoodClassifierApp()
    
    # Try to load a sample image from the dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_classes = ['pizza', 'hamburger', 'sushi', 'ice_cream', 'chocolate_cake']
    
    for food_class in sample_classes:
        image_dir = os.path.join(script_dir, 'data', 'images', food_class)
        
        if os.path.exists(image_dir):
            # Get the first image from this class
            images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
            
            if images:
                image_path = os.path.join(image_dir, images[0])
                
                try:
                    # Load and classify the image
                    image = Image.open(image_path)
                    print(f"\n📸 Testing {food_class.replace('_', ' ').title()}:")
                    print(f"   Image: {images[0]}")
                    
                    # Get predictions
                    results = classifier.predict(image, top_k=3)
                    
                    if results:
                        print("   Predictions:")
                        for i, (pred_class, confidence) in enumerate(results, 1):
                            formatted_name = pred_class.replace('_', ' ').title()
                            print(f"   {i}. {formatted_name}: {confidence*100:.2f}%")
                        
                        # Check if prediction is correct
                        top_pred = results[0][0]
                        is_correct = top_pred == food_class
                        print(f"   Correct: {'✅' if is_correct else '❌'}")
                    else:
                        print("   ❌ Prediction failed")
                        
                except Exception as e:
                    print(f"   ❌ Error: {e}")
            else:
                print(f"\n📸 No images found for {food_class}")
        else:
            print(f"\n📸 Directory not found for {food_class}")
    
    print("\n" + "=" * 40)
    print("🎉 Demo completed!")
    print("💡 To run the web app, use: python run_app.py")

if __name__ == "__main__":
    demo_classification()
