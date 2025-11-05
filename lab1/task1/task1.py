import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = ...
model = ... # remember about cache_dir="/tmp/"
tokenizer = ... # remember about cache_dir="/tmp/"

print(f"Using device: {device}")
print(f"Loading model: {model_name}")
print(f"Tokenizer loaded with vocabulary size: {len(tokenizer)}")
print(f"Model loaded successfully with {sum(p.numel() for p in model.parameters())} parameters")


def get_embedding(text, tokenizer, model):
    ...
    return embedding.cpu().numpy()

def cosine(a, b):
    ...
    return similarity

sentence1 = "The cat sat on the mat."
sentence2 = "A cat is sitting on a mat."
sentence3 = "Dogs are playing in the park."

#print(f"Sentence 1: {sentence1}")
#print(f"Sentence 2: {sentence2}")
#print(f"Sentence 3: {sentence3}")

embedding1 = get_embedding(sentence1, tokenizer, model)
embedding2 = get_embedding(sentence2, tokenizer, model)
embedding3 = get_embedding(sentence3, tokenizer, model)

##print(f"Embedding shapes: {embedding1.shape}, {embedding2.shape}, {embedding3.shape}")

sim_1_2 = cosine(embedding1, embedding2)
sim_1_3 = cosine(embedding1, embedding3)
sim_2_3 = cosine(embedding2, embedding3)

#print(f"Cosine similarity between sentence 1 and 2: {sim_1_2:.4f}")
#print(f"Cosine similarity between sentence 1 and 3: {sim_1_3:.4f}")
#print(f"Cosine similarity between sentence 2 and 3: {sim_2_3:.4f}")
