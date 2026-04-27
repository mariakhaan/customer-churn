"""
Model Trainer Module

Trains and evaluates multiple classification models.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
import joblib
import json
from typing import Dict, Tuple, Any
import os


class ModelTrainer:
    """Train and evaluate churn prediction models."""
    
    def __init__(self, config: Dict):
        """
        Initialize model trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models = {}
        self.results = {}
        
    def initialize_models(self) -> None:
        """Initialize all enabled models from config."""
        model_config = self.config['models']
        
        if model_config['logistic_regression']['enabled']:
            self.models['Logistic Regression'] = LogisticRegression(
                **model_config['logistic_regression']['params']
            )
            
        if model_config['random_forest']['enabled']:
            self.models['Random Forest'] = RandomForestClassifier(
                **model_config['random_forest']['params']
            )
            
        if model_config['xgboost']['enabled']:
            self.models['XGBoost'] = XGBClassifier(
                **model_config['xgboost']['params']
            )
        
        print(f"Initialized {len(self.models)} models")
    
    def train_model(
        self, 
        model_name: str, 
        model: Any, 
        X_train: np.ndarray, 
        y_train: np.ndarray
    ) -> Any:
        """
        Train a single model.
        
        Args:
            model_name: Name of the model
            model: Model instance
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
        """
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        print(f"{model_name} training complete")
        return model
    
    def evaluate_model(
        self,
        model_name: str,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            model_name: Name of the model
            model: Trained model instance
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n{model_name} Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"  TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
        print(f"  FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")
        
        return metrics
    
    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> None:
        """
        Train and evaluate all models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
        """
        self.initialize_models()
        
        for model_name, model in self.models.items():
            # Train
            trained_model = self.train_model(model_name, model, X_train, y_train)
            
            # Evaluate
            metrics = self.evaluate_model(model_name, trained_model, X_test, y_test)
            
            # Store results
            self.results[model_name] = metrics
            
            # Save model
            self.save_model(trained_model, model_name)
    
    def save_model(self, model: Any, model_name: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            model: Trained model
            model_name: Name of the model
        """
        save_path = self.config['training']['model_save_path']
        os.makedirs(save_path, exist_ok=True)
        
        filename = f"{save_path}{model_name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, filename)
        print(f"Model saved to {filename}")
    
    def save_metrics(self) -> None:
        """Save evaluation metrics to JSON file."""
        metrics_file = self.config['output']['metrics_file']
        
        with open(metrics_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nMetrics saved to {metrics_file}")
    
    def get_best_model(self) -> Tuple[str, Dict[str, float]]:
        """
        Get the best performing model based on ROC-AUC.
        
        Returns:
            Tuple of (model_name, metrics)
        """
        best_model = max(self.results.items(), key=lambda x: x[1]['roc_auc'])
        return best_model
    
    def print_summary(self) -> None:
        """Print a summary of all model results."""
        print("\n" + "="*70)
        print("MODEL COMPARISON SUMMARY")
        print("="*70)
        
        # Create comparison DataFrame
        df_results = pd.DataFrame(self.results).T
        df_results = df_results.round(4)
        
        print(df_results.to_string())
        
        best_model_name, best_metrics = self.get_best_model()
        print(f"\n{'='*70}")
        print(f"BEST MODEL: {best_model_name}")
        print(f"ROC-AUC: {best_metrics['roc_auc']:.4f}")
        print("="*70)


def main():
    """Test model training pipeline."""
    import yaml
    from preprocessing import DataPreprocessor
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Prepare data
    preprocessor = DataPreprocessor(config['preprocessing'])
    X_train, X_test, y_train, y_test = preprocessor.prepare_train_test(
        config['data']['output_path']
    )
    
    # Train models
    trainer = ModelTrainer(config)
    trainer.train_all(X_train, y_train, X_test, y_test)
    trainer.save_metrics()
    trainer.print_summary()


if __name__ == "__main__":
    main()