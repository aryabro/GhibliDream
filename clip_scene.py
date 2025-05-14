import torch
import torch.nn.functional as torch_nn_func
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation, CLIPTokenizer
from PIL import Image
import os
import numpy as np
import faiss
import glob

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model and processor
CLIPSEG_CACHE_DIR = os.getenv('CLIPSEG_CACHE_DIR')
if not CLIPSEG_CACHE_DIR:
    raise ValueError("CLIPSEG_CACHE_DIR environment variable not set")

model = CLIPSegForImageSegmentation.from_pretrained('CIDAS/clipseg-rd64-refined', cache_dir=CLIPSEG_CACHE_DIR).to(device)
processor = CLIPSegProcessor.from_pretrained('CIDAS/clipseg-rd64-refined', cache_dir=CLIPSEG_CACHE_DIR)
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32', cache_dir=CLIPSEG_CACHE_DIR)

# Define prompts
prompts = ["a scene"]  # Single prompt for scene comparison

# Get all images from the images folder
image_paths = sorted(glob.glob('images/*.jpg'))
if not image_paths:
    raise ValueError("No images found in the images folder")

print(f"Found {len(image_paths)} images to process")

embeddings_list = []
index = None

for image_path in image_paths:
    # Load image
    pil_image = Image.open(image_path).convert('RGB')
    width, height = pil_image.size

    # Prepare inputs
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=77)
    image_inputs = processor(images=[pil_image], return_tensors="pt")
    inputs.update(image_inputs)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get and normalize image embeddings
    image_embeddings = outputs.vision_model_output.pooler_output
    image_embeddings = torch_nn_func.normalize(image_embeddings, p=2, dim=1)
    
    # Convert to numpy and store
    embeddings_np = image_embeddings.cpu().numpy()
    embeddings_list.append(embeddings_np)
    
    # Initialize FAISS index with first embedding's dimension
    if index is None:
        dimension = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dimension)
    
    # Add to FAISS index
    index.add(embeddings_np)
    
    # Get segmentation predictions
    preds = torch.sigmoid(outputs.logits)
    preds_resized = torch_nn_func.interpolate(preds.unsqueeze(1), size=(height, width), 
                                             mode="bilinear", align_corners=False).squeeze(1)
    
    print(f"\nProcessing {image_path}:")
    print(f"Image embeddings shape: {image_embeddings.shape}")
    print(f"Predictions shape: {preds_resized.shape}")
    print(f"Predictions min/max: {preds_resized.min()}/{preds_resized.max()}")

# Compare first image against all others
query_embedding = embeddings_list[0]
distances, indices = index.search(query_embedding, k=len(image_paths))
# TODO add other similarity measures 

# Convert distances to similarities (assuming L2 distance)
max_dist = np.max(distances)
similarities = 1 - (distances / max_dist)

# Print results
print("\nSimilarity results (comparing first image against all others):")
for i, (dist, idx, sim) in enumerate(zip(distances[0], indices[0], similarities[0])):
    print(f"Image 1 vs Image {idx+1}:")
    print(f"  Distance: {dist:.4f}")
    print(f"  Similarity: {sim:.4f}")
    print(f"  Image path: {image_paths[idx]}")
    print() 