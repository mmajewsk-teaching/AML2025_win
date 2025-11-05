import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent plots from showing
import matplotlib.pyplot as plt
import os
import inspect
from sklearn.manifold import TSNE

# Import the relevant functions and variables from the student's file
try:
    from task2 import get_embeddings, tokenizer, model, texts, reduced_embeddings, tsne
except ImportError as e:
    print(f"Error importing from task2.py: {e}")
    raise

class TestLab2Task(unittest.TestCase):
    
    def test_model_tokenizer_loaded(self):
        """Test that model and tokenizer are properly loaded"""
        self.assertIsNotNone(tokenizer, "Tokenizer not loaded")
        self.assertIsNotNone(model, "Model not loaded")
        
        # Check if model is actually a model from transformers
        self.assertTrue(str(type(model)).find('transformers') > -1, 
                        "Model should be a transformers model")
        
        # Check if tokenizer is actually a tokenizer from transformers
        self.assertTrue(str(type(tokenizer)).find('transformers') > -1, 
                        "Tokenizer should be a transformers tokenizer")
        
    def test_get_embeddings(self):
        """Test that get_embeddings function works correctly"""
        # Test with a simple text
        sample_text = ["This is a test sentence."]
        embeddings = get_embeddings(sample_text, tokenizer, model)
        
        # Check that we get the expected shape (1 text, model dimension)
        self.assertEqual(len(embeddings.shape), 2, "Embeddings should be 2-dimensional")
        self.assertEqual(embeddings.shape[0], 1, "Should have 1 embedding for 1 text")
        self.assertEqual(embeddings.shape[1], 384, "Embedding dimension should be 384 for the MiniLM model")
        
    def test_tsne_used(self):
        """Test that t-SNE class was actually used"""
        # Check that tsne is an instance of sklearn's TSNE
        self.assertIsInstance(tsne, TSNE, "tsne should be an instance of sklearn.manifold.TSNE")
        
        # Check that t-SNE parameters are set properly
        self.assertEqual(tsne.n_components, 2, "t-SNE should use 2 components for 2D visualization")
        
    def test_tsne_reduction(self):
        """Test that t-SNE reduction was applied correctly"""
        # Check that we have the right shape for reduced embeddings
        self.assertEqual(reduced_embeddings.shape, (len(texts), 2), 
                         "Reduced embeddings should have shape (n_texts, 2)")
        
        # Check that the embeddings were actually transformed (not just zeros)
        self.assertFalse(np.allclose(reduced_embeddings, 0), 
                         "Reduced embeddings should not be all zeros")
        
    def test_visualization_saved(self):
        """Test that the visualization was created and saved"""
        # Check if the figure file exists
        self.assertTrue(os.path.exists("tsne_visualization.png"), 
                        "Visualization file should be saved")
        self.assertTrue(os.path.getsize("tsne_visualization.png") > 0, 
                        "Visualization file should not be empty")

if __name__ == '__main__':
    unittest.main()