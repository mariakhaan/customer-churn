"""
Predictor Module

Makes predictions on new customer data using trained models.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Union, List, Dict
import os


class ChurnPredictor:
    """Make churn predictions using trained models."""
    
    def __init__(self, model_path: str):
        """
        Initialize predictor with a trained model.
        
        Args:
            model_path: Path to saved model file
        """
        self.model = joblib.load(model_path)
        self.model_name = os.path.basename(model_path).replace('.pkl', '')
        print(f"Loaded model: {self.model_name}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict churn labels.
        
        Args:
            X: Feature array
            
        Returns:
            Predicted labels (0 or 1)
        """
        predictions = self.model.predict(X)
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict churn probabilities.
        
        Args:
            X: Feature array
            
        Returns:
            Predicted probabilities for class 1 (churn)
        """
        probabilities = self.model.predict_proba(X)[:, 1]
        return probabilities
    
    def predict_single(
        self, 
        X: np.ndarray, 
        threshold: float = 0.5
    ) -> Dict[str, Union[int, float, str]]:
        """
        Predict for a single customer with detailed output.
        
        Args:
            X: Feature array for single customer (1D or 2D)
            threshold: Classification threshold
            
        Returns:
            Dictionary with prediction details
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        proba = self.predict_proba(X)[0]
        prediction = 1 if proba >= threshold else 0
        
        result = {
            'will_churn': bool(prediction),
            'churn_probability': float(proba),
            'confidence': float(max(proba, 1 - proba)),
            'risk_level': self._get_risk_level(proba)
        }
        
        return result
    
    def _get_risk_level(self, probability: float) -> str:
        """
        Categorize churn risk based on probability.
        
        Args:
            probability: Churn probability
            
        Returns:
            Risk level string
        """
        if probability < 0.3:
            return "Low"
        elif probability < 0.6:
            return "Medium"
        elif probability < 0.8:
            return "High"
        else:
            return "Very High"
    
    def batch_predict(
        self, 
        X: np.ndarray, 
        customer_ids: List[str] = None
    ) -> pd.DataFrame:
        """
        Make predictions for multiple customers.
        
        Args:
            X: Feature array
            customer_ids: Optional list of customer IDs
            
        Returns:
            DataFrame with predictions
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        results = pd.DataFrame({
            'customer_id': customer_ids if customer_ids else range(len(X)),
            'will_churn': predictions,
            'churn_probability': probabilities,
            'risk_level': [self._get_risk_level(p) for p in probabilities]
        })
        
        return results
    
    def get_high_risk_customers(
        self,
        X: np.ndarray,
        customer_ids: List[str] = None,
        threshold: float = 0.7
    ) -> pd.DataFrame:
        """
        Identify high-risk customers.
        
        Args:
            X: Feature array
            customer_ids: Optional list of customer IDs
            threshold: Probability threshold for high risk
            
        Returns:
            DataFrame of high-risk customers
        """
        results = self.batch_predict(X, customer_ids)
        high_risk = results[results['churn_probability'] >= threshold]
        high_risk = high_risk.sort_values('churn_probability', ascending=False)
        
        return high_risk


def main():
    """Test predictor with sample data."""
    # Load a trained model (XGBoost by default)
    model_path = "models/xgboost.pkl"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train models first by running: python main.py")
        return
    
    predictor = ChurnPredictor(model_path)
    
    # Create sample customer data (scaled features)
    # In practice, this would come from preprocessed new data
    sample_customer = np.array([[
        15.0,   # tenure
        75.5,   # monthly_charges
        1100.0, # total_charges
        0,      # contract (encoded)
        1,      # payment_method (encoded)
        1,      # internet_service (encoded)
        0,      # tech_support (encoded)
        3,      # num_services
        73.3,   # avg_monthly_charge
        0,      # tenure_group (encoded)
        1,      # charge_group (encoded)
        3       # service_score
    ]])
    
    # Single prediction
    result = predictor.predict_single(sample_customer)
    
    print("\nSample Customer Prediction:")
    print(f"  Will Churn: {result['will_churn']}")
    print(f"  Probability: {result['churn_probability']:.2%}")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Confidence: {result['confidence']:.2%}")


if __name__ == "__main__":
    main()