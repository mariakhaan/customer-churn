"""
Preprocessing Module

Handles data cleaning, feature engineering, and transformation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict


class DataPreprocessor:
    """Preprocess customer data for churn prediction."""
    
    def __init__(self, config: Dict):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Dictionary containing preprocessing configuration
        """
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Loaded DataFrame
        """
        df = pd.read_csv(filepath)
        print(f"Loaded data: {df.shape}")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Remove customer_id (not a feature)
        if 'customer_id' in df_clean.columns:
            df_clean = df_clean.drop('customer_id', axis=1)
        
        # Handle missing values
        df_clean = df_clean.dropna()
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Ensure total_charges is positive
        df_clean['total_charges'] = df_clean['total_charges'].clip(lower=0)
        
        print(f"Cleaned data: {df_clean.shape}")
        return df_clean
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing ones.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df_feat = df.copy()
        
        # Average monthly charge (total/tenure)
        df_feat['avg_monthly_charge'] = df_feat['total_charges'] / (df_feat['tenure'] + 1)
        
        # Tenure categories
        df_feat['tenure_group'] = pd.cut(
            df_feat['tenure'], 
            bins=[0, 12, 24, 48, 100], 
            labels=['0-1yr', '1-2yr', '2-4yr', '4yr+']
        ).astype(object)
        
        # Charge categories
        df_feat['charge_group'] = pd.cut(
            df_feat['monthly_charges'],
            bins=[0, 40, 70, 100, 200],
            labels=['low', 'medium', 'high', 'very_high']
        ).astype(object)
        
        # Service engagement score
        df_feat['service_score'] = (
            (df_feat['tech_support'] == 'Yes').astype(int) + 
            df_feat['num_services']
        )
        
        return df_feat
    
    def encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit encoders (True for training, False for inference)
            
        Returns:
            DataFrame with encoded features
        """
        df_encoded = df.copy()
        
        # Get categorical columns
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != self.config['target']]
        
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    le = self.label_encoders[col]
                    df_encoded[col] = df_encoded[col].map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
        
        return df_encoded
    
    def split_features_target(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split features and target variable.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features, target)
        """
        target_col = self.config['target']
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Scale numerical features.
        
        Args:
            X: Feature DataFrame
            fit: Whether to fit the scaler
            
        Returns:
            Scaled features as numpy array
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def prepare_train_test(
        self, 
        filepath: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline for training.
        
        Args:
            filepath: Path to data file
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Load and clean
        df = self.load_data(filepath)
        df = self.clean_data(df)
        
        # Feature engineering
        df = self.engineer_features(df)
        
        # Encode categorical features
        df = self.encode_features(df, fit=True)
        
        # Split features and target
        X, y = self.split_features_target(df)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scale_features(X_train, fit=True)
        X_test_scaled = self.scale_features(X_test, fit=False)
        
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        print(f"Churn rate (train): {y_train.mean():.2%}")
        print(f"Churn rate (test): {y_test.mean():.2%}")
        
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values


def main():
    """Test preprocessing pipeline."""
    config = {
        'target': 'churn',
        'test_size': 0.2,
        'random_state': 42
    }
    
    preprocessor = DataPreprocessor(config)
    X_train, X_test, y_train, y_test = preprocessor.prepare_train_test(
        "data/customer_data.csv"
    )
    
    print(f"\nFeature names: {preprocessor.feature_names}")


if __name__ == "__main__":
    main()