import unittest
import numpy as np
import torch
from task1 import cosine, get_embedding, tokenizer, model

class TestLab1Functions(unittest.TestCase):
    
    def test_cosine_simple_cases(self):
        # Test with orthogonal vectors
        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        self.assertEqual(cosine(a, b), 0.0)
        
        # Test with identical vectors
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3])
        self.assertAlmostEqual(cosine(a, b), 1.0)
        
        # Test with opposite vectors
        a = np.array([1, 2, 3])
        b = np.array([-1, -2, -3])
        self.assertAlmostEqual(cosine(a, b), -1.0)
        
        # Test with specific vectors
        a = np.array([0.2, 0.3, 0.5])
        b = np.array([0.1, 0.4, 0.5])
        expected = 0.9762210399274298
        self.assertAlmostEqual(cosine(a, b), expected)
    
    def test_get_embedding(self):
        # Test with actual example sentences
        test_sentence1 = "The cat sat on the mat."
        test_sentence2 = "A cat is sitting on a mat."
        
        # Get embeddings
        emb1 = get_embedding(test_sentence1, tokenizer, model)
        emb2 = get_embedding(test_sentence2, tokenizer, model)
        
        # Check shape
        self.assertEqual(emb1.shape, (384,))
        self.assertEqual(emb2.shape, (384,))
        
        # Check first few values match expected pattern
        # These values come from the actual output of the model
        self.assertAlmostEqual(emb1[0], 0.7842203, places=5)
        self.assertAlmostEqual(emb1[1], -0.09497593, places=5)
        self.assertAlmostEqual(emb2[0], 0.6081358, places=5)
        self.assertAlmostEqual(emb2[1], -0.24017286, places=5)
        
        # Check cosine similarity between related sentences
        similarity = cosine(emb1, emb2)
        self.assertAlmostEqual(similarity, 0.9372, places=4)

    def test_real_sentences(self):
        # Define test sentences
        sentence1 = "The cat sat on the mat."
        sentence2 = "A cat is sitting on a mat."
        sentence3 = "Dogs are playing in the park."
        
        # Get embeddings
        embedding1 = get_embedding(sentence1, tokenizer, model)
        embedding2 = get_embedding(sentence2, tokenizer, model)
        embedding3 = get_embedding(sentence3, tokenizer, model)
        
        # Calculate similarity scores
        sim_1_2 = cosine(embedding1, embedding2)
        sim_1_3 = cosine(embedding1, embedding3)
        sim_2_3 = cosine(embedding2, embedding3)
        
        # Check similarity values match expected values
        self.assertAlmostEqual(sim_1_2, 0.9372, places=4)
        self.assertAlmostEqual(sim_1_3, 0.0101, places=4)
        self.assertAlmostEqual(sim_2_3, 0.0184, places=4)
        
        # Verify that semantically similar sentences have higher similarity
        self.assertGreater(sim_1_2, sim_1_3)
        self.assertGreater(sim_1_2, sim_2_3)

if __name__ == '__main__':
    unittest.main()
