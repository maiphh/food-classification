"""
Batch processing script for food classification
Processes multiple images in a directory and saves results to CSV
"""

import os
import sys
import argparse
import pandas as pd
from PIL import Image
from pathlib import Path

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from app import FoodClassifierApp

def process_images_in_directory(input_dir, output_file=None, top_k=3):
    """
    Process all images in a directory and save results.
    
    Args:
        input_dir (str): Directory containing images to classify
        output_file (str): Output CSV file path (optional)
        top_k (int): Number of top predictions to save
    """
    # Initialize classifier
    print("🤖 Initializing Food Classifier...")
    classifier = FoodClassifierApp()
    
    # Find all image files
    input_path = Path(input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"❌ No image files found in {input_dir}")
        return
    
    print(f"📁 Found {len(image_files)} images to process")
    
    # Process each image
    results = []
    
    for i, image_file in enumerate(image_files, 1):
        print(f"📸 Processing {i}/{len(image_files)}: {image_file.name}")
        
        try:
            # Load and classify image
            image = Image.open(image_file)
            predictions = classifier.predict(image, top_k=top_k)
            
            if predictions:
                # Store results
                for rank, (food_class, confidence) in enumerate(predictions, 1):
                    results.append({
                        'filename': image_file.name,
                        'filepath': str(image_file),
                        'rank': rank,
                        'predicted_class': food_class,
                        'formatted_class': food_class.replace('_', ' ').title(),
                        'confidence': confidence,
                        'confidence_percent': f"{confidence*100:.2f}%"
                    })
            else:
                print(f"   ❌ Failed to classify {image_file.name}")
                
        except Exception as e:
            print(f"   ❌ Error processing {image_file.name}: {e}")
    
    if not results:
        print("❌ No successful classifications")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Generate output filename if not provided
    if output_file is None:
        output_file = f"food_classification_results_{len(image_files)}_images.csv"
    
    # Save results
    df.to_csv(output_file, index=False)
    print(f"💾 Results saved to: {output_file}")
    
    # Show summary
    print("\n📊 Summary:")
    print(f"   Total images processed: {len(image_files)}")
    print(f"   Successful classifications: {len(df['filename'].unique())}")
    print(f"   Top predictions per image: {top_k}")
    
    # Show top predicted classes
    top_predictions = df[df['rank'] == 1]['formatted_class'].value_counts().head(10)
    if not top_predictions.empty:
        print(f"\n🏆 Most common top predictions:")
        for food_class, count in top_predictions.items():
            print(f"   {food_class}: {count} images")

def main():
    """Main function for batch processing."""
    parser = argparse.ArgumentParser(description='Batch Food Classification')
    parser.add_argument('input_dir', help='Directory containing images to classify')
    parser.add_argument('--output', '-o', help='Output CSV file path')
    parser.add_argument('--top-k', '-k', type=int, default=3,
                       help='Number of top predictions per image (default: 3)')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        return
    
    if not os.path.isdir(args.input_dir):
        print(f"❌ Input path is not a directory: {args.input_dir}")
        return
    
    print("🍕 Food Classification Batch Processor")
    print("=" * 50)
    print(f"📁 Input directory: {args.input_dir}")
    print(f"📊 Top predictions per image: {args.top_k}")
    
    if args.output:
        print(f"💾 Output file: {args.output}")
    
    print("=" * 50)
    
    # Process images
    process_images_in_directory(args.input_dir, args.output, args.top_k)
    
    print("\n🎉 Batch processing completed!")

if __name__ == "__main__":
    main()
