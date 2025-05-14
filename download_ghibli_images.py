import kaggle
import random
from pathlib import Path
import shutil

def download_and_copy_images():
    # Create directories
    output_dir = Path("ghibli_images")
    images_dir = Path("images")
    
    # Clean up existing directories
    if output_dir.exists():
        shutil.rmtree(output_dir)
    if images_dir.exists():
        shutil.rmtree(images_dir)
    
    output_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    
    try:
        # Download dataset
        print("Downloading dataset...")
        kaggle.api.dataset_download_files(
            'amulbel/ghibli-movies-pictures',
            path=output_dir,
            unzip=True,
            force=True
        )
        
        # Get movie directories
        movie_dirs = [d for d in (output_dir / "Ghibli_movie_dataset").iterdir() if d.is_dir()]
        print(f"Found {len(movie_dirs)} movie directories")
        
        # Copy 3 random images from each movie
        for movie_dir in movie_dirs:
            print(f"\nProcessing {movie_dir.name}...")
            images = list(movie_dir.glob("*.jpg")) + list(movie_dir.glob("*.png"))
            
            if not images:
                print(f"No images found in {movie_dir.name}")
                continue
            
            # Select and copy 3 random images
            selected_images = random.sample(images, min(3, len(images)))
            
            for i, img in enumerate(selected_images, 1):
                # Create new filename with movie name
                new_name = f"{movie_dir.name}_{i}{img.suffix}"
                dest = images_dir / new_name
                shutil.copy2(img, dest)
                print(f"Copied {new_name}")
        
        print("\nDone! Images have been downloaded and copied to the 'images' directory.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if images_dir.exists():
            shutil.rmtree(images_dir)
        raise

if __name__ == "__main__":
    download_and_copy_images() 