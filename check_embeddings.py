import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

def get_clip_embeddings(text_prompts: list[str], image_paths: list[str], model_name: str = "openai/clip-vit-base-patch32"):
    """
    Computes text embeddings for given text prompts and image embeddings for given image paths
    using a specified CLIP model from the transformers library.

    Args:
        text_prompts (list[str]): A list of text strings.
        image_paths (list[str]): A list of file paths to images.
        model_name (str): The name of the pre-trained CLIP model to use
                          (e.g., "openai/clip-vit-base-patch32").

    Returns:
        tuple: A tuple containing two elements:
               - text_embeddings (torch.Tensor or None): A tensor of text embeddings.
                 None if text_prompts is empty or None.
               - image_embeddings (torch.Tensor or None): A tensor of image embeddings.
                 None if image_paths is empty or None.
    """
    try:
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model or processor: {e}")
        print("Please ensure you have 'transformers', 'torch', and 'Pillow' installed,")
        print("and the model name is correct (e.g., 'openai/clip-vit-base-patch32').")
        print("You might need to install them: pip install transformers torch Pillow")
        return None, None

    text_embeddings = None
    if text_prompts:
        try:
            inputs = processor(text=text_prompts, return_tensors="pt", padding=True, truncation=True)
            text_outputs = model.get_text_features(**inputs)
            text_embeddings = text_outputs
        except Exception as e:
            print(f"Error processing text prompts: {e}")

    image_embeddings = None
    if image_paths:
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

        if images:
            try:
                inputs = processor(text=None, images=images, return_tensors="pt", padding=True)
                image_outputs = model.get_image_features(**inputs)
                image_embeddings = image_outputs
            except Exception as e:
                print(f"Error processing images: {e}")
        elif image_paths: # if image_paths was not empty but images is (all failed to load)
             print("No valid images were processed.")


    return text_embeddings, image_embeddings

if __name__ == "__main__":
    # Example Usage:
    # 1. Create dummy image files for testing
    try:
        for i in range(2):
            img = Image.new('RGB', (60, 30), color = ('red' if i == 0 else 'blue'))
            img.save(f"dummy_image_{i+1}.png")
        print("Created dummy_image_1.png and dummy_image_2.png for example.")
    except ImportError:
        print("Pillow (PIL) is not installed. Cannot create dummy images for the example.")
        print("Please install it: pip install Pillow")
    except Exception as e:
        print(f"Could not create dummy images: {e}")


    sample_texts = ["a photo of a cat", "a drawing of a dog"]
    sample_image_files = ["dummy_image_1.png", "dummy_image_2.png", "non_existent_image.png"]

    print(f"\nProcessing texts: {sample_texts}")
    print(f"Processing images: {sample_image_files}\n")

    # It's good practice to specify the device if you have a GPU
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model.to(device)
    # And ensure tensors are moved to the same device

    text_embeds, img_embeds = get_clip_embeddings(sample_texts, sample_image_files)

    if text_embeds is not None:
        print("\nText Embeddings:")
        print(text_embeds)
        print(f"Shape: {text_embeds.shape}")

    if img_embeds is not None:
        print("\nImage Embeddings:")
        print(img_embeds)
        print(f"Shape: {img_embeds.shape}")

    # Clean up dummy images
    import os
    try:
        os.remove("dummy_image_1.png")
        os.remove("dummy_image_2.png")
        print("\nCleaned up dummy images.")
    except OSError:
        pass # Files might not have been created if Pillow was missing 