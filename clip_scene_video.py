import torch
import torch.nn.functional as torch_nn_func
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation, CLIPTokenizer
from PIL import Image
import os
import numpy as np
import faiss
import cv2
import matplotlib.pyplot as plt
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

# Get all video files
video_paths = sorted(glob.glob('images/r2b_*.mp4'))
video_paths = sorted(glob.glob('images/input3.mp4'))
if not video_paths:
    raise ValueError("No video files found matching pattern 'images/r2b_*.mp4'")

print(f"Found {len(video_paths)} videos to process")

def process_video(video_path):
    print(f"\nProcessing video: {video_path}")
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video file: {video_path}")
        return

    embeddings_list = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
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
        
        frame_count += 1
        if frame_count % 10 == 0:  # Print progress every 10 frames
            print(f"Processed frame {frame_count}")

    cap.release()

    if frame_count == 0:
        print(f"No frames processed for {video_path}")
        return

    # Compare first frame against all others
    query_embedding = embeddings_list[0]
    dimension = embeddings_np.shape[1]

    # L2 Distance (Euclidean)
    l2_index = faiss.IndexFlatL2(dimension)
    l2_index.add(np.vstack(embeddings_list))
    l2_distances, l2_indices = l2_index.search(query_embedding, k=frame_count)

    # Cosine Similarity (using normalized vectors and inner product)
    normalized_embeddings = np.vstack(embeddings_list)
    faiss.normalize_L2(normalized_embeddings)
    normalized_query = query_embedding.copy()
    faiss.normalize_L2(normalized_query)
    cosine_index = faiss.IndexFlatIP(dimension)
    cosine_index.add(normalized_embeddings)
    cosine_similarities, cosine_indices = cosine_index.search(normalized_query, k=frame_count)

    # Manhattan Distance (L1)
    def manhattan_distance(a, b):
        return np.sum(np.abs(a - b), axis=1)

    manhattan_distances = np.array([
        manhattan_distance(query_embedding, emb) for emb in embeddings_list
    ]).reshape(1, -1)

    # Dot Product
    def dot_product(a, b):
        return np.sum(a * b, axis=1)

    dot_products = np.array([
        dot_product(query_embedding, emb) for emb in embeddings_list
    ]).reshape(1, -1)

    # Convert all measures to similarity scores [0,1]
    def distance_to_similarity(distances):
        max_dist = np.max(distances)
        if max_dist == 0:
            return np.ones_like(distances)
        return 1 - (distances / max_dist)

    l2_similarity = distance_to_similarity(l2_distances[0])
    manhattan_similarity = distance_to_similarity(manhattan_distances[0])

    # Normalize dot products to [0,1]
    dot_similarity = (dot_products[0] - np.min(dot_products[0])) / (np.max(dot_products[0]) - np.min(dot_products[0]))

    # Create plots
    plt.figure(figsize=(15, 8))

    # Plot all similarity measures in one graph
    plt.plot(range(frame_count), l2_similarity, 'b-', label='L2 Similarity')
    plt.plot(range(frame_count), cosine_similarities[0], 'r-', label='Cosine Similarity')
    plt.plot(range(frame_count), manhattan_similarity, 'g-', label='Manhattan Similarity')
    plt.plot(range(frame_count), dot_similarity, 'm-', label='Dot Product Similarity')

    # Add normalized versions
    def normalize_to_01(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x)) if np.max(x) > np.min(x) else np.ones_like(x)

    # plt.plot(range(frame_count), normalize_to_01(l2_similarity), 'b--', alpha=0.5, label='L2 Similarity (normalized)')
    # plt.plot(range(frame_count), normalize_to_01(cosine_similarities[0]), 'r--', alpha=0.5, label='Cosine Similarity (normalized)')
    # plt.plot(range(frame_count), normalize_to_01(manhattan_similarity), 'g--', alpha=0.5, label='Manhattan Similarity (normalized)')
    # plt.plot(range(frame_count), normalize_to_01(dot_similarity), 'm--', alpha=0.5, label='Dot Product Similarity (normalized)')

    plt.title(f'Frame Similarity Measures Over Time\n{os.path.basename(video_path)}')
    plt.xlabel('Frame Number')
    plt.ylabel('Similarity Score')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save plot with video filename
    output_filename = f'similarity_analysis_{os.path.splitext(os.path.basename(video_path))[0]}.png'
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"\nPlot saved as '{output_filename}'")
    plt.close()

# Process all videos
for video_path in video_paths:
    process_video(video_path)

print("\nProcessing complete!") 