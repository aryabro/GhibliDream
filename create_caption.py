import os
from pathlib import Path
import google.generativeai as genai
from PIL import Image

# Configure the Gemini API
# GOOGLE_API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key
GOOGLE_API_KEY = "TODO"  # Replace with your actual API key

# generate api key here https://ai.google.dev/gemini-api/docs/quickstart?lang=python#make-first-request
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the Gemini Pro Vision model
model = genai.GenerativeModel('gemini-2.0-flash')

def generate_caption(image_path, prompt="Describe this image in detail"):
    """Generate a caption for the given image using Gemini Pro Vision."""
    try:
        # Load the image
        image = Image.open(image_path)
        
        # Generate caption
        response = model.generate_content([prompt, image])
        
        # Extract the caption text
        caption = response.text
        return caption
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def process_directory(directory_path, prompt="Describe this image in detail"):
    """Process all images in the directory and create caption files."""
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    
    # Convert to Path object
    directory = Path(directory_path)
    
    # Process each image in the directory
    for image_path in directory.iterdir():
        if image_path.suffix.lower() in image_extensions:
            print(f"Processing {image_path.name}...")
            
            # Generate caption
            caption = generate_caption(image_path, prompt)
            
            if caption:
                # Create caption file with same name but .txt extension
                caption_file = image_path.with_suffix('.txt')
                
                # Write caption to file
                with open(caption_file, 'w', encoding='utf-8') as f:
                    f.write(caption)
                
                print(f"Created caption file: {caption_file.name}")

if __name__ == "__main__":
    # Hardcoded directory path
    image_directory = "images"  # Replace with your actual directory path
    prompt = "Describe this image in detail"
    
    
    # Process the directory
    process_directory(image_directory, prompt)
