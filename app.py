"""
Linear Regression Model Application
This module trains and serves a linear regression model for predictions.
"""

import json
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import numpy as np
import pickle
import os


class LinearRegressionModel:
    """Linear Regression Model Wrapper"""
    
    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False
        self.model_path = "model.pkl"
    
    def train(self, X, y):
        """
        Train the linear regression model.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training target values (n_samples,)
        """
        self.model.fit(X, y)
        self.is_trained = True
        print(f"Model trained successfully. Coefficient: {self.model.coef_}, Intercept: {self.model.intercept_}")
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Features for prediction (n_samples, n_features)
            
        Returns:
            Predictions (n_samples,)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """
        Evaluate model performance using R² score.
        
        Args:
            X: Test features
            y: Test target values
            
        Returns:
            R² score
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        return self.model.score(X, y)
    
    def save(self, path=None):
        """Save model to disk"""
        save_path = path or self.model_path
        with open(save_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {save_path}")
    
    def load(self, path=None):
        """Load model from disk"""
        load_path = path or self.model_path
        if os.path.exists(load_path):
            with open(load_path, 'rb') as f:
                self.model = pickle.load(f)
            self.is_trained = True
            print(f"Model loaded from {load_path}")
        else:
            raise FileNotFoundError(f"Model file not found at {load_path}")


def create_sample_data():
    """Generate sample training data"""
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    return X, y


def main():
    """Main function to demonstrate the model"""
    print("=" * 50)
    print("Linear Regression Model Training")
    print("=" * 50)
    
    # Create and train model
    model = LinearRegressionModel()
    X, y = create_sample_data()
    
    # Split data (simple 80-20 split)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train model
    model.train(X_train, y_train)
    
    # Evaluate
    score = model.evaluate(X_test, y_test)
    print(f"R² Score: {score:.4f}")
    
    # Make predictions
    predictions = model.predict(X_test[:5])
    print(f"\nSample predictions: {predictions}")
    print(f"Actual values: {y_test[:5]}")
    
    # Save model
    model.save()
    print("\nModel training complete!")


if __name__ == "__main__":
    main()
