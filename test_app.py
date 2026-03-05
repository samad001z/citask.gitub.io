"""
Test cases for Linear Regression Model
"""

import unittest
import numpy as np
import tempfile
import os
from app import LinearRegressionModel, create_sample_data


class TestLinearRegressionModel(unittest.TestCase):
    """Test cases for LinearRegressionModel class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = LinearRegressionModel()
        self.X, self.y = create_sample_data()
        
        # Create simple test data
        self.X_train = np.array([[1], [2], [3], [4], [5]])
        self.y_train = np.array([2, 4, 6, 8, 10])  # y = 2x
        
        self.X_test = np.array([[6], [7]])
        self.y_test = np.array([12, 14])
    
    def test_model_initialization(self):
        """Test that model initializes correctly"""
        self.assertFalse(self.model.is_trained)
        self.assertEqual(self.model.model_path, "model.pkl")
    
    def test_train_model(self):
        """Test training functionality"""
        self.model.train(self.X_train, self.y_train)
        self.assertTrue(self.model.is_trained)
        
        # Check coefficients are close to expected (2x relationship)
        self.assertAlmostEqual(self.model.model.coef_[0], 2.0, places=5)
    
    def test_predict_before_training(self):
        """Test that predict raises error before training"""
        with self.assertRaises(ValueError):
            self.model.predict(self.X_test)
    
    def test_predict_after_training(self):
        """Test prediction after training"""
        self.model.train(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        # Check predictions are close to expected
        np.testing.assert_array_almost_equal(predictions, self.y_test, decimal=0)
    
    def test_evaluate_before_training(self):
        """Test that evaluate raises error before training"""
        with self.assertRaises(ValueError):
            self.model.evaluate(self.X_test, self.y_test)
    
    def test_evaluate_after_training(self):
        """Test evaluation after training"""
        self.model.train(self.X_train, self.y_train)
        score = self.model.evaluate(self.X_test, self.y_test)
        
        # Score should be high for this simple linear relationship
        self.assertGreater(score, 0.9)
        self.assertLessEqual(score, 1.0)
    
    def test_save_and_load(self):
        """Test model saving and loading"""
        self.model.train(self.X_train, self.y_train)
        
        # Use temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            tmp_path = tmp.name
        
        try:
            # Save model
            self.model.save(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Create new model and load
            new_model = LinearRegressionModel()
            new_model.load(tmp_path)
            
            # Verify loaded model produces same predictions
            predictions_original = self.model.predict(self.X_test)
            predictions_loaded = new_model.predict(self.X_test)
            
            np.testing.assert_array_almost_equal(predictions_original, predictions_loaded)
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises error"""
        with self.assertRaises(FileNotFoundError):
            self.model.load("nonexistent_file.pkl")
    
    def test_model_coefficients(self):
        """Test model coefficients after training"""
        self.model.train(self.X_train, self.y_train)
        
        # Coefficients should be accessible
        self.assertIsNotNone(self.model.model.coef_)
        self.assertIsNotNone(self.model.model.intercept_)
    
    def test_prediction_shape(self):
        """Test that prediction shape matches input"""
        self.model.train(self.X, self.y)
        
        # Test single prediction
        single = self.model.predict(self.X[:1])
        self.assertEqual(single.shape, (1,))
        
        # Test multiple predictions
        multiple = self.model.predict(self.X[:10])
        self.assertEqual(multiple.shape, (10,))
    
    def test_training_with_sample_data(self):
        """Test training with generated sample data"""
        split_idx = int(len(self.X) * 0.8)
        X_train, X_test = self.X[:split_idx], self.X[split_idx:]
        y_train, y_test = self.y[:split_idx], self.y[split_idx:]
        
        self.model.train(X_train, y_train)
        score = self.model.evaluate(X_test, y_test)
        
        # Should have reasonable score
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == '__main__':
    unittest.main()
