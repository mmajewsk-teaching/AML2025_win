import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
# We'll need to import TSNE from scikit-learn
from sklearn.manifold import TSNE  # Add this import

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use the same model as in lab2_notes
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = ... # remember about cache_dir="/tmp/"
model = ... # remember about cache_dir="/tmp/"

print(f"Using device: {device}")
print(f"Loading model: {model_name}")

# Sample texts - similar to what we used in lab2_notes
texts = [
    "Artificial intelligence is revolutionizing many fields.",
    "Deep learning models require large amounts of data.",
    "PyTorch is a popular framework for machine learning.",
    "Transformers have improved natural language processing significantly.",
    "Retrieval augmented generation enhances language models with external knowledge."
]

# Function to get embeddings from our model
def get_embeddings(texts, tokenizer, model):
    ...
    return embeddings.cpu().numpy()

# Get embeddings for our texts
embeddings = ...

# Create a t-SNE instance
# Use 2 components for 2D visualization
tsne = ...

# Apply t-SNE to our embeddings
reduced_embeddings = ...

# Create a visualization of the reduced embeddings
plt.figure(figsize=(8, 6))
plt.s...(..., ..., marker='o', s=100)
for i, text in enumerate(texts):
    plt.annotate(text, (..., ..., fontsize=8)
plt.title("t-SNE Visualization of Text Embeddings")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.tight_layout()

# Save the figure
plt.savefig("tsne_visualization.png")
plt.show()
