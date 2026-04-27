"""
Main Pipeline

Run the complete churn prediction workflow:
1. Generate synthetic data
2. Preprocess and engineer features
3. Train multiple models
4. Evaluate and compare results
"""

import yaml
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_generator import ChurnDataGenerator
from src.preprocessing import DataPreprocessor
from src.model_trainer import ModelTrainer


def main():
    """Run the complete pipeline."""
    print("="*70)
    print("CUSTOMER CHURN PREDICTION PIPELINE")
    print("="*70)
    
    # Load configuration
    print("\n[1/4] Loading configuration...")
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("✓ Configuration loaded")
    
    # Generate data
    print("\n[2/4] Generating synthetic customer data...")
    os.makedirs('data', exist_ok=True)
    generator = ChurnDataGenerator(
        n_samples=config['data']['n_samples'],
        random_state=config['data']['random_state']
    )
    df = generator.generate()
    generator.save(df, config['data']['output_path'])
    print("✓ Data generation complete")
    
    # Preprocess data
    print("\n[3/4] Preprocessing and feature engineering...")
    preprocessor_config = {
    **config['preprocessing'],
    'test_size': config['data']['test_size'],
    'random_state': config['data']['random_state'],
    }
    preprocessor = DataPreprocessor(preprocessor_config)
    X_train, X_test, y_train, y_test = preprocessor.prepare_train_test(
        config['data']['output_path']
    )
    print("✓ Preprocessing complete")
    
    # Train models
    print("\n[4/4] Training and evaluating models...")
    os.makedirs('models', exist_ok=True)
    trainer = ModelTrainer(config)
    trainer.train_all(X_train, y_train, X_test, y_test)
    trainer.save_metrics()
    trainer.print_summary()
    print("\n✓ Pipeline complete!")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("• View results in: models/metrics.json")
    print("• Run tests: pytest tests/")
    print("• Make predictions: python src/predictor.py")


if __name__ == "__main__":
    main()