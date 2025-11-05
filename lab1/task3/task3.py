import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/tmp/")
model = AutoModel.from_pretrained(model_name, cache_dir="/tmp/").to(device)

print(f"Using device: {device}")
print(f"Loading model: {model_name}")

texts = [
    "Artificial intelligence is revolutionizing many fields.",
    "Deep learning models require large amounts of data.",
    "PyTorch is a popular framework for machine learning.",
    "Transformers have improved natural language processing significantly.",
    "Retrieval augmented generation enhances language models with external knowledge."
]

def get_embeddings(texts, tokenizer, model):
    ...
    return embeddings.cpu().numpy()

embeddings = get_embeddings(texts, tokenizer, model)
print(f"Generated embeddings with shape: {embeddings.shape}")

query = "How do neural networks learn from data?"
query_embedding = get_embeddings([...], ..., ...)[0]
print(f"Generated query embedding with shape: {query_embedding.shape}")

# Implement three similarity metrics:

# 1. Cosine similarity (already familiar from previous labs)
def cosine_similarity(a, b):
    return ...

# 2. Dot product similarity
def dot_product_similarity(a, b):
    return ...

# 3. Euclidean distance similarity (convert distance to similarity)
def euclidean_similarity(a, b):
    return ...

# Function to find most similar texts using specified metric
def find_similar_texts(query_embedding, text_embeddings, similarity_func):
    similarities = []
    for i, ... in enumerate(...):
        similarity = ...(..., ...)
        similarities.append((i, ...))
    # Sort by similarity score (highest first)
    return ...(similarities, key=lambda x: ..., reverse=True)

# Compare results using different similarity metrics
cosine_results = find_similar_texts(query_embedding, embeddings, ...)
dot_product_results = find_similar_texts(query_embedding, embeddings, ...)
euclidean_results = find_similar_texts(query_embedding, embeddings, ...)

# Prepare data for visualization
similarity_data = {
    "Cosine": [score for _, score in cosine_results],
    "Dot Product": [score for _, score in dot_product_results],
    "Euclidean": [score for _, score in euclidean_results]
}

# Get original text order for each ranking
text_rankings = {
    "Cosine": [idx for idx, _ in cosine_results],
    "Dot Product": [idx for idx, _ in dot_product_results],
    "Euclidean": [idx for idx, _ in euclidean_results]
}

# Create bar chart comparison
plt.figure(figsize=(15, 8))

# Set up the bar positions
bar_width = 0.25
index = np.arange(len(texts))

plt.bar(index - bar_width, ..., bar_width, label="Cosine")
plt.bar(index, ..., bar_width, label="Dot Product")
plt.bar(index + bar_width, ..., bar_width, label="Euclidean")

plt.xlabel("Text")
plt.ylabel("Similarity Score")
plt.title(f"Comparison of Similarity Metrics for Query: '{query}'")
plt.xticks(index, [f"Text {i+1}" for i in range(len(texts))])
plt.legend()
plt.tight_layout()

# Save the figure
plt.savefig("similarity_comparison.png")
plt.show()

# Create a second visualization showing the ranking differences
plt.figure(figsize=(12, 6))

ranking_matrix = np.zeros((len(texts), 3))
for i, metric_idx in enumerate(["Cosine", "Dot Product", "Euclidean"]):
    for rank, text_idx in enumerate(text_rankings[metric_idx]):
        ranking_matrix[text_idx, i] = rank + 1  # Store 1-based rank


sns.heatmap(ranking_matrix, annot=True, cmap="YlGnBu", 
            xticklabels=["Cosine", "Dot Product", "Euclidean"],
            yticklabels=[f"Text {i+1}" for i in range(len(texts))],
            cbar_kws={'label': 'Rank (1 = most similar)'})

plt.title("Ranking Comparison Between Similarity Metrics")
plt.tight_layout()

# Save the figure
plt.savefig("ranking_comparison.png")
plt.show()

print("Results for Query:", query)
for metric_name in ["Cosine", "Dot Product", "Euclidean"]:
    print(f"\n{metric_name} Similarity Results:")
    for i, idx in enumerate(text_rankings[metric_name]):
        print(f"{i+1}. Text {idx+1}: {texts[idx][:50]}... (Score: {similarity_data[metric_name][i]:.4f})")
