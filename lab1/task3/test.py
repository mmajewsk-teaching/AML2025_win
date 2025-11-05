import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent plots from showing
import matplotlib.pyplot as plt
import os

# Import the relevant functions and variables from the student's file
try:
    from task3 import (
        cosine_similarity,
        dot_product_similarity,
        euclidean_similarity,
        find_similar_texts,
        model,
        tokenizer,
        embeddings,
        query_embedding
    )
except ImportError as e:
    print(f"Error importing from task3.py: {e}")
    raise

class TestLab3Task(unittest.TestCase):
    
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
    
    def test_cosine_similarity(self):
        """Test cosine similarity implementation"""
        # Test with parallel vectors
        a = np.array([1, 0, 0])
        b = np.array([2, 0, 0])
        self.assertAlmostEqual(cosine_similarity(a, b), 1.0, 
                              msg="Cosine similarity of parallel vectors should be 1.0")
        
        # Test with orthogonal vectors
        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        self.assertAlmostEqual(cosine_similarity(a, b), 0.0, 
                              msg="Cosine similarity of orthogonal vectors should be 0.0")
        
        # Test with opposite vectors
        a = np.array([1, 0, 0])
        b = np.array([-1, 0, 0])
        self.assertAlmostEqual(cosine_similarity(a, b), -1.0, 
                              msg="Cosine similarity of opposite vectors should be -1.0")
        
        # Test with actual embeddings
        if embeddings.shape[0] >= 2:
            sim = cosine_similarity(embeddings[0], embeddings[1])
            self.assertTrue(-1.0 <= sim <= 1.0, 
                          msg="Cosine similarity should be between -1 and 1")
    
    def test_dot_product_similarity(self):
        """Test dot product similarity implementation"""
        # Test with simple vectors
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        self.assertAlmostEqual(dot_product_similarity(a, b), 32.0, 
                              msg="Dot product should be sum of element-wise products")
        
        # Test with orthogonal vectors
        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        self.assertAlmostEqual(dot_product_similarity(a, b), 0.0, 
                              msg="Dot product of orthogonal vectors should be 0.0")
        
        # Test with actual embeddings
        if embeddings.shape[0] >= 2:
            dot_prod = dot_product_similarity(embeddings[0], embeddings[1])
            self.assertTrue(isinstance(dot_prod, (int, float, np.number)), 
                          msg="Dot product should return a number")
    
    def test_euclidean_similarity(self):
        """Test euclidean similarity implementation"""
        # Test with identical vectors
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3])
        self.assertAlmostEqual(euclidean_similarity(a, b), 1.0, 
                              msg="Euclidean similarity of identical vectors should be 1.0")
        
        # Test with distant vectors
        a = np.array([0, 0, 0])
        b = np.array([10, 10, 10])
        # For very distant vectors, similarity should approach 0
        self.assertTrue(euclidean_similarity(a, b) < 0.1, 
                        msg="Euclidean similarity of distant vectors should be close to 0")
        
        # Test with actual embeddings
        if embeddings.shape[0] >= 2:
            eucl_sim = euclidean_similarity(embeddings[0], embeddings[1])
            self.assertTrue(0.0 <= eucl_sim <= 1.0, 
                          msg="Euclidean similarity should be between 0 and 1")
    
    def test_find_similar_texts(self):
        """Test the find_similar_texts function"""
        # Create some test embeddings
        test_embeddings = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0]
        ])
        test_query = np.array([1, 1, 0])
        
        # Test with cosine similarity
        results = find_similar_texts(test_query, test_embeddings, cosine_similarity)
        
        # Check if the results are sorted by similarity (highest first)
        similarities = [sim for _, sim in results]
        self.assertEqual(len(similarities), len(test_embeddings), 
                         "Should return similarity for each embedding")
        self.assertTrue(all(similarities[i] >= similarities[i+1] for i in range(len(similarities)-1)), 
                       "Results should be sorted by similarity (highest first)")
        
        # Check if the most similar vector to [1,1,0] is itself or [1,1,0]
        most_similar_idx = results[0][0]
        self.assertTrue(
            np.array_equal(test_embeddings[most_similar_idx], test_query) or 
            cosine_similarity(test_embeddings[most_similar_idx], test_query) == 1.0,
            "Most similar vector to query should be itself or have similarity 1.0"
        )

    def test_output_files_created(self):
        """Test that visualization files were created"""
        self.assertTrue(os.path.exists("similarity_comparison.png"), 
                       "similarity_comparison.png file should be created")
        self.assertTrue(os.path.getsize("similarity_comparison.png") > 0, 
                       "similarity_comparison.png should not be empty")
        
        self.assertTrue(os.path.exists("ranking_comparison.png"), 
                       "ranking_comparison.png file should be created")
        self.assertTrue(os.path.getsize("ranking_comparison.png") > 0, 
                       "ranking_comparison.png should not be empty")

if __name__ == '__main__':
    unittest.main()