import torch
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel
from PIL import Image

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

if __name__ == "__main__":
    # Example Usage:
    dummy_image_files = []
    try:
        for i in range(2):
            filename = f"dummy_image_{i+1}.png"
            img = Image.new('RGB', (60, 30), color = ('red' if i == 0 else 'blue'))
            img.save(filename)
            dummy_image_files.append(filename)
        print(f"Created {dummy_image_files} for example.")
    except ImportError:
        print("Pillow (PIL) is not installed. Cannot create dummy images for the example.")
        print("Please install it: pip install Pillow")
    except Exception as e:
        print(f"Could not create dummy images: {e}")

    sample_texts = ["a photo of a cat", "a drawing of a dog"]
    # Add a non-existent image to test error handling
    sample_image_paths = dummy_image_files + ["non_existent_image.png"]


    print(f"\nProcessing texts: {sample_texts}")
    text_embeds = get_clip_text_embeddings(sample_texts)
    if text_embeds is not None:
        print("\nText Embeddings (CLIP):")
        print(f"Shape: {text_embeds.shape}")

    print(f"\nProcessing images: {sample_image_paths}")
    img_embeds = get_clip_image_embeddings(sample_image_paths)
    if img_embeds is not None:
        print("\nImage Embeddings (CLIP):")
        print(f"Shape: {img_embeds.shape}")

    # Example with pre-loading model and processor for CLIP
    print("\n--- Example with pre-loaded CLIP model/processor ---")
    model_id = "openai/clip-vit-base-patch32"
    try:
        shared_model = CLIPModel.from_pretrained(model_id)
        shared_processor = CLIPProcessor.from_pretrained(model_id)
        
        # Optional: Move model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        shared_model.to(device)
        print(f"Using CLIP device: {device}")

        text_embeds_shared = get_clip_text_embeddings(
            sample_texts, model=shared_model, processor=shared_processor
        )
        if text_embeds_shared is not None:
            print("\nText Embeddings (CLIP pre-loaded model):")
            print(f"Shape: {text_embeds_shared.shape}")

        img_embeds_shared = get_clip_image_embeddings(
            dummy_image_files, model=shared_model, processor=shared_processor # using only valid images here
        )
        if img_embeds_shared is not None:
            print("\nImage Embeddings (CLIP pre-loaded model):")
            print(f"Shape: {img_embeds_shared.shape}")

    except Exception as e:
        print(f"Error in CLIP pre-loading example: {e}")

    print("\n--- Example with DINOv2 ---")
    # Using only valid dummy images for DINOv2 example to keep output cleaner
    if dummy_image_files: # Only run if dummy images were created
        print(f"\nProcessing images with DINOv2: {dummy_image_files}")
        dinov2_embeds = get_dinov2_image_embeddings(dummy_image_files)
        if dinov2_embeds is not None:
            print("\nImage Embeddings (DINOv2):")
            print(f"Shape: {dinov2_embeds.shape}")

        # Example with pre-loading DINOv2 model and processor
        print("\n--- Example with pre-loaded DINOv2 model/processor ---")
        dinov2_model_id = "facebook/dinov2-base"
        try:
            shared_dinov2_processor = AutoImageProcessor.from_pretrained(dinov2_model_id)
            shared_dinov2_model = AutoModel.from_pretrained(dinov2_model_id)

            # Optional: Move model to GPU if available
            device_dino = "cuda" if torch.cuda.is_available() else "cpu"
            shared_dinov2_model.to(device_dino)
            print(f"Using DINOv2 device: {device_dino}")

            dinov2_embeds_shared = get_dinov2_image_embeddings(
                dummy_image_files, 
                model=shared_dinov2_model, 
                processor=shared_dinov2_processor
            )
            if dinov2_embeds_shared is not None:
                print("\nImage Embeddings (DINOv2 pre-loaded model):")
                print(f"Shape: {dinov2_embeds_shared.shape}")

        except Exception as e:
            print(f"Error in DINOv2 pre-loading example: {e}")
    else:
        print("\nSkipping DINOv2 example as dummy images could not be created.")


    # Clean up dummy images
    import os
    for fname in dummy_image_files:
        try:
            os.remove(fname)
        except OSError:
            pass # File might not have been created 