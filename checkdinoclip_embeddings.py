import torch
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def _load_model_and_processor(model_name: str, model=None, processor=None):
    """Helper function to load model and processor if not provided."""
    if model is None or processor is None:
        try:
            loaded_model = CLIPModel.from_pretrained(model_name)
            loaded_processor = CLIPProcessor.from_pretrained(model_name)
            if torch.cuda.is_available():
                device = torch.device("cuda")
                loaded_model.to(device)
            return loaded_model, loaded_processor
        except Exception as e:
            print(f"Error loading model or processor ({model_name}): {e}")
            print("Please ensure you have 'transformers', 'torch', and 'Pillow' installed,")
            print("and the model name is correct (e.g., 'openai/clip-vit-base-patch32').")
            print("You might need to install them: pip install transformers torch Pillow")
            raise # Re-raise the exception to be caught by the calling function
    return model, processor

def _load_dinov2_model_and_processor(model_name: str, model=None, processor=None):
    """Helper function to load DINOv2 model and processor if not provided."""
    if model is None or processor is None:
        try:
            loaded_processor = AutoImageProcessor.from_pretrained(model_name)
            loaded_model = AutoModel.from_pretrained(model_name)
            if torch.cuda.is_available():
                device = torch.device("cuda")
                loaded_model.to(device)
            return loaded_model, loaded_processor
        except Exception as e:
            print(f"Error loading DINOv2 model or processor ({model_name}): {e}")
            print("Please ensure you have 'transformers', 'torch', and 'Pillow' installed,")
            print("and the model name is correct (e.g., 'facebook/dinov2-base').")
            print("You might need to install them: pip install transformers torch Pillow")
            raise # Re-raise the exception to be caught by the calling function
    return model, processor

def get_clip_text_embeddings(
    text_prompts: list[str],
    model_name: str = "openai/clip-vit-base-patch32",
    model: CLIPModel = None,
    processor: CLIPProcessor = None
):
    """
    Computes text embeddings for given text prompts using a specified CLIP model.

    Args:
        text_prompts (list[str]): A list of text strings.
        model_name (str): The name of the pre-trained CLIP model to use if model/processor not provided.
        model (CLIPModel, optional): Pre-loaded CLIPModel.
        processor (CLIPProcessor, optional): Pre-loaded CLIPProcessor.

    Returns:
        torch.Tensor or None: A tensor of text embeddings, or None if an error occurs or no prompts.
    """
    if not text_prompts:
        print("No text prompts provided.")
        return None

    try:
        model, processor = _load_model_and_processor(model_name, model, processor)
    except Exception:
        return None

    try:
        inputs = processor(text=text_prompts, return_tensors="pt", padding=True, truncation=True)
        device = model.device # Ensure inputs are on the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        text_outputs = model.get_text_features(**inputs)
        return text_outputs
    except Exception as e:
        print(f"Error processing text prompts: {e}")
        return None

def get_clip_image_embeddings(
    image_paths: list[str],
    model_name: str = "openai/clip-vit-base-patch32",
    model: CLIPModel = None,
    processor: CLIPProcessor = None
):
    """
    Computes image embeddings for given image paths using a specified CLIP model.

    Args:
        image_paths (list[str]): A list of file paths to images.
        model_name (str): The name of the pre-trained CLIP model to use if model/processor not provided.
        model (CLIPModel, optional): Pre-loaded CLIPModel.
        processor (CLIPProcessor, optional): Pre-loaded CLIPProcessor.

    Returns:
        torch.Tensor or None: A tensor of image embeddings, or None if an error occurs or no valid images.
    """
    if not image_paths:
        print("No image paths provided.")
        return None

    try:
        model, processor = _load_model_and_processor(model_name, model, processor)
    except Exception:
        return None

    images = []
    valid_image_paths = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
            valid_image_paths.append(path)
        except FileNotFoundError:
            print(f"Warning: Image file not found at {path}")
        except Exception as e:
            print(f"Warning: Could not open or process image {path}: {e}")

    if not images:
        print("No valid images were loaded.")
        return None

    try:
        inputs = processor(text=None, images=images, return_tensors="pt", padding=True)
        device = model.device # Ensure inputs are on the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        image_outputs = model.get_image_features(**inputs)
        return image_outputs
    except Exception as e:
        print(f"Error processing images: {e}")
        return None

def get_dinov2_image_embeddings(
    image_paths: list[str],
    model_name: str = "facebook/dinov2-base",
    model: AutoModel = None,
    processor: AutoImageProcessor = None
):
    """
    Computes image embeddings for given image paths using a specified DINOv2 model.

    Args:
        image_paths (list[str]): A list of file paths to images.
        model_name (str): The name of the pre-trained DINOv2 model to use 
                          (e.g., "facebook/dinov2-base") if model/processor not provided.
        model (AutoModel, optional): Pre-loaded DINOv2 AutoModel.
        processor (AutoImageProcessor, optional): Pre-loaded DINOv2 AutoImageProcessor.

    Returns:
        torch.Tensor or None: A tensor of image embeddings, or None if an error occurs or no valid images.
    """
    if not image_paths:
        print("No image paths provided for DINOv2.")
        return None

    try:
        model, processor = _load_dinov2_model_and_processor(model_name, model, processor)
    except Exception:
        return None

    images_pil = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images_pil.append(img)
        except FileNotFoundError:
            print(f"Warning (DINOv2): Image file not found at {path}")
        except Exception as e:
            print(f"Warning (DINOv2): Could not open or process image {path}: {e}")

    if not images_pil:
        print("No valid images were loaded for DINOv2.")
        return None

    try:
        inputs = processor(images=images_pil, return_tensors="pt")
        
        # Move inputs to the same device as the model
        device = model.device 
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad(): # DINOv2 is usually used for feature extraction, no grad needed
            outputs = model(**inputs)
        
        # DINOv2 typically returns 'last_hidden_state'.
        # We will take the CLS token embedding, which is the first token in the sequence.
        # The output shape is (batch_size, sequence_length, hidden_size)
        last_hidden_state = outputs.last_hidden_state
        # CLS token is at index 0 of the sequence_length dimension
        embeddings = last_hidden_state[:, 0]
        
        return embeddings
    except Exception as e:
        print(f"Error processing images with DINOv2: {e}")
        return None

def process_numbered_images_from_folders(
    model_name: str = "openai/clip-vit-base-patch32",
    dinov2_model_name: str = "facebook/dinov2-base"
):
    """
    Process images from specific folders and create embeddings for corresponding numbered images.
    Folders: input_faces, input_backgrounds, merge1token, merge2token
    Images should be numbered 1,2,3,4 in each folder.
    
    Args:
        model_name (str): CLIP model name
        dinov2_model_name (str): DINOv2 model name
        
    Returns:
        dict: Dictionary containing embeddings organized by folder name and image number
    """
    import os
    from pathlib import Path
    
    # Define folders to process
    folders = ['input_faces', 'input_backgrounds', 'merge1token', 'merge2token']
    results = {}
    
    # Load models once
    clip_model, clip_processor = _load_model_and_processor(model_name)
    dinov2_model, dinov2_processor = _load_dinov2_model_and_processor(dinov2_model_name)
    
    # Initialize results structure
    for folder in folders:
        results[folder] = {
            'clip_embeddings': {},
            'dinov2_embeddings': {},
            'image_paths': {}
        }
    
    # Process each number (1-4)
    for num in range(1, 5):
        for folder in folders:
            # Try both jpg and png extensions
            image_path = None
            for ext in ['.jpg', '.png']:
                path = Path(folder) / f"{num}{ext}"
                if path.exists():
                    image_path = str(path)
                    break
            
            if image_path:
                # Get CLIP embeddings
                clip_embeds = get_clip_image_embeddings(
                    [image_path],
                    model=clip_model,
                    processor=clip_processor
                )
                
                # Get DINOv2 embeddings
                dinov2_embeds = get_dinov2_image_embeddings(
                    [image_path],
                    model=dinov2_model,
                    processor=dinov2_processor
                )
                
                # Store results
                results[folder]['clip_embeddings'][num] = clip_embeds
                results[folder]['dinov2_embeddings'][num] = dinov2_embeds
                results[folder]['image_paths'][num] = image_path
    
    return results

def compute_cosine_similarity(emb1, emb2):
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        emb1 (torch.Tensor): First embedding
        emb2 (torch.Tensor): Second embedding
        
    Returns:
        float: Cosine similarity score
    """
    # Normalize the embeddings
    emb1_norm = emb1 / emb1.norm(dim=-1, keepdim=True)
    emb2_norm = emb2 / emb2.norm(dim=-1, keepdim=True)
    
    # Compute cosine similarity
    similarity = torch.mm(emb1_norm, emb2_norm.t())
    return similarity.item()

def compute_l2_distance(emb1, emb2):
    """
    Compute L2 (Euclidean) distance between two embeddings.
    
    Args:
        emb1 (torch.Tensor): First embedding
        emb2 (torch.Tensor): Second embedding
        
    Returns:
        float: L2 distance score
    """
    return torch.norm(emb1 - emb2).item()

def compute_manhattan_distance(emb1, emb2):
    """
    Compute Manhattan (L1) distance between two embeddings.
    
    Args:
        emb1 (torch.Tensor): First embedding
        emb2 (torch.Tensor): Second embedding
        
    Returns:
        float: Manhattan distance score
    """
    return torch.sum(torch.abs(emb1 - emb2)).item()

def distance_to_similarity(distance, max_distance=None):
    """
    Convert a distance score to a similarity score in range [0,1].
    If max_distance is None, uses the provided distance as max.
    
    Args:
        distance (float): Distance score
        max_distance (float, optional): Maximum possible distance
        
    Returns:
        float: Similarity score in range [0,1]
    """
    if max_distance is None:
        max_distance = distance
    if max_distance == 0:
        return 1.0
    return 1.0 - (distance / max_distance)

def plot_similarities(results, model_type='clip', merge_folder='merge1token'):
    """
    Plot various similarity metrics between folders.
    
    Args:
        results (dict): Results dictionary from process_numbered_images_from_folders
        model_type (str): Either 'clip' or 'dinov2'
        merge_folder (str): Either 'merge1token' or 'merge2token'
    """
    # Get image numbers that exist in both folders
    numbers = []
    bg_cosine_sims = []
    face_cosine_sims = []
    bg_l2_sims = []
    face_l2_sims = []
    bg_manhattan_sims = []
    face_manhattan_sims = []
    
    # Track maximum distances for normalization
    max_l2_distance = 0
    max_manhattan_distance = 0
    
    # First pass: compute all distances and find maximums
    for num in range(1, 5):
        if (num in results['input_backgrounds'][f'{model_type}_embeddings'] and 
            num in results[merge_folder][f'{model_type}_embeddings'] and
            num in results['input_faces'][f'{model_type}_embeddings']):
            
            # Background distances
            bg_l2 = compute_l2_distance(
                results['input_backgrounds'][f'{model_type}_embeddings'][num],
                results[merge_folder][f'{model_type}_embeddings'][num]
            )
            bg_manhattan = compute_manhattan_distance(
                results['input_backgrounds'][f'{model_type}_embeddings'][num],
                results[merge_folder][f'{model_type}_embeddings'][num]
            )
            
            # Face distances
            face_l2 = compute_l2_distance(
                results['input_faces'][f'{model_type}_embeddings'][num],
                results[merge_folder][f'{model_type}_embeddings'][num]
            )
            face_manhattan = compute_manhattan_distance(
                results['input_faces'][f'{model_type}_embeddings'][num],
                results[merge_folder][f'{model_type}_embeddings'][num]
            )
            
            max_l2_distance = max(max_l2_distance, bg_l2, face_l2)
            max_manhattan_distance = max(max_manhattan_distance, bg_manhattan, face_manhattan)
    
    # Second pass: compute similarities and store results
    for num in range(1, 5):
        if (num in results['input_backgrounds'][f'{model_type}_embeddings'] and 
            num in results[merge_folder][f'{model_type}_embeddings'] and
            num in results['input_faces'][f'{model_type}_embeddings']):
            
            numbers.append(num)
            
            # Background similarities
            bg_cosine = compute_cosine_similarity(
                results['input_backgrounds'][f'{model_type}_embeddings'][num],
                results[merge_folder][f'{model_type}_embeddings'][num]
            )
            bg_l2 = compute_l2_distance(
                results['input_backgrounds'][f'{model_type}_embeddings'][num],
                results[merge_folder][f'{model_type}_embeddings'][num]
            )
            bg_manhattan = compute_manhattan_distance(
                results['input_backgrounds'][f'{model_type}_embeddings'][num],
                results[merge_folder][f'{model_type}_embeddings'][num]
            )
            
            # Face similarities
            face_cosine = compute_cosine_similarity(
                results['input_faces'][f'{model_type}_embeddings'][num],
                results[merge_folder][f'{model_type}_embeddings'][num]
            )
            face_l2 = compute_l2_distance(
                results['input_faces'][f'{model_type}_embeddings'][num],
                results[merge_folder][f'{model_type}_embeddings'][num]
            )
            face_manhattan = compute_manhattan_distance(
                results['input_faces'][f'{model_type}_embeddings'][num],
                results[merge_folder][f'{model_type}_embeddings'][num]
            )
            
            # Convert distances to similarities
            bg_l2_sim = distance_to_similarity(bg_l2, max_l2_distance)
            face_l2_sim = distance_to_similarity(face_l2, max_l2_distance)
            bg_manhattan_sim = distance_to_similarity(bg_manhattan, max_manhattan_distance)
            face_manhattan_sim = distance_to_similarity(face_manhattan, max_manhattan_distance)
            
            # Store results
            bg_cosine_sims.append(bg_cosine)
            face_cosine_sims.append(face_cosine)
            bg_l2_sims.append(bg_l2_sim)
            face_l2_sims.append(face_l2_sim)
            bg_manhattan_sims.append(bg_manhattan_sim)
            face_manhattan_sims.append(face_manhattan_sim)
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    x = np.arange(len(numbers))
    width = 0.15  # Reduced width to accommodate more bars
    
    # Plot all metrics
    plt.bar(x - width*2, bg_cosine_sims, width, label='Background Cosine')
    plt.bar(x - width, bg_l2_sims, width, label='Background L2')
    plt.bar(x, bg_manhattan_sims, width, label='Background Manhattan')
    plt.bar(x + width, face_cosine_sims, width, label='Face Cosine')
    plt.bar(x + width*2, face_l2_sims, width, label='Face L2')
    plt.bar(x + width*3, face_manhattan_sims, width, label='Face Manhattan')
    
    plt.xlabel('Image Number')
    plt.ylabel('Similarity Score')
    plt.title(f'{model_type.upper()} Embedding Similarities with {merge_folder}')
    plt.xticks(x, numbers)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on top of bars
    for i, v in enumerate(bg_cosine_sims):
        plt.text(i - width*2, v, f'{v:.3f}', ha='center', va='bottom', rotation=45)
    for i, v in enumerate(bg_l2_sims):
        plt.text(i - width, v, f'{v:.3f}', ha='center', va='bottom', rotation=45)
    for i, v in enumerate(bg_manhattan_sims):
        plt.text(i, v, f'{v:.3f}', ha='center', va='bottom', rotation=45)
    for i, v in enumerate(face_cosine_sims):
        plt.text(i + width, v, f'{v:.3f}', ha='center', va='bottom', rotation=45)
    for i, v in enumerate(face_l2_sims):
        plt.text(i + width*2, v, f'{v:.3f}', ha='center', va='bottom', rotation=45)
    for i, v in enumerate(face_manhattan_sims):
        plt.text(i + width*3, v, f'{v:.3f}', ha='center', va='bottom', rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot with appropriate name
    plt.savefig(f'{model_type}_similarities_{merge_folder}.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Process numbered images from folders
    print("\n--- Processing numbered images from folders ---")
    try:
        results = process_numbered_images_from_folders()
        
        # Compare embeddings between folders
        print("\n--- Comparing embeddings between folders ---")
        
        # Compare with merge1token
        print("\nComparing with merge1token:")
        for num in range(1, 5):
            if (num in results['input_backgrounds']['clip_embeddings'] and 
                num in results['merge1token']['clip_embeddings']):
                print(f"\nImage number {num}:")
                # CLIP similarity
                clip_sim = compute_cosine_similarity(
                    results['input_backgrounds']['clip_embeddings'][num],
                    results['merge1token']['clip_embeddings'][num]
                )
                print(f"CLIP similarity: {clip_sim:.4f}")
                
                # DINOv2 similarity
                dinov2_sim = compute_cosine_similarity(
                    results['input_backgrounds']['dinov2_embeddings'][num],
                    results['merge1token']['dinov2_embeddings'][num]
                )
                print(f"DINOv2 similarity: {dinov2_sim:.4f}")
        
        # Compare with merge2token
        print("\nComparing with merge2token:")
        for num in range(1, 5):
            if (num in results['input_backgrounds']['clip_embeddings'] and 
                num in results['merge2token']['clip_embeddings']):
                print(f"\nImage number {num}:")
                # CLIP similarity
                clip_sim = compute_cosine_similarity(
                    results['input_backgrounds']['clip_embeddings'][num],
                    results['merge2token']['clip_embeddings'][num]
                )
                print(f"CLIP similarity: {clip_sim:.4f}")
                
                # DINOv2 similarity
                dinov2_sim = compute_cosine_similarity(
                    results['input_backgrounds']['dinov2_embeddings'][num],
                    results['merge2token']['dinov2_embeddings'][num]
                )
                print(f"DINOv2 similarity: {dinov2_sim:.4f}")
        
        # Create and save plots for both merge folders
        print("\nCreating similarity plots...")
        # CLIP plots
        plot_similarities(results, 'clip', 'merge1token')
        plot_similarities(results, 'clip', 'merge2token')
        # DINOv2 plots
        plot_similarities(results, 'dinov2', 'merge1token')
        plot_similarities(results, 'dinov2', 'merge2token')
        print("Plots saved as:")
        print("- clip_similarities_merge1token.png")
        print("- clip_similarities_merge2token.png")
        print("- dinov2_similarities_merge1token.png")
        print("- dinov2_similarities_merge2token.png")
        
        # Print original results
        print("\n--- Original results ---")
        for folder in results:
            print(f"\nResults for folder: {folder}")
            for num in range(1, 5):
                if num in results[folder]['image_paths']:
                    print(f"\n  Image number {num}:")
                    print(f"  Image path: {results[folder]['image_paths'][num]}")
                    if results[folder]['clip_embeddings'][num] is not None:
                        print(f"  CLIP embeddings shape: {results[folder]['clip_embeddings'][num].shape}")
                    if results[folder]['dinov2_embeddings'][num] is not None:
                        print(f"  DINOv2 embeddings shape: {results[folder]['dinov2_embeddings'][num].shape}")
    except Exception as e:
        print(f"Error processing numbered images: {e}") 