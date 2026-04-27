"""
Unit Tests for Preprocessing Module

Tests data cleaning, feature engineering, and encoding functions.
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import DataPreprocessor


@pytest.fixture
def sample_data():
    """Create sample customer data for testing."""
    return pd.DataFrame({
        'customer_id': ['CUST_001', 'CUST_002', 'CUST_003', 'CUST_004'],
        'tenure': [12, 24, 6, 36],
        'monthly_charges': [50.0, 75.0, 100.0, 45.0],
        'total_charges': [600.0, 1800.0, 600.0, 1620.0],
        'contract': ['Month-to-month', 'One year', 'Month-to-month', 'Two year'],
        'payment_method': ['Electronic check', 'Bank transfer', 'Credit card', 'Mailed check'],
        'internet_service': ['DSL', 'Fiber optic', 'Fiber optic', 'No'],
        'tech_support': ['No', 'Yes', 'No', 'No internet service'],
        'num_services': [2, 5, 3, 1],
        'churn': [1, 0, 1, 0]
    })


@pytest.fixture
def preprocessor():
    """Create preprocessor instance."""
    config = {
        'target': 'churn',
        'test_size': 0.2,
        'random_state': 42
    }
    return DataPreprocessor(config)


class TestDataCleaning:
    """Test data cleaning functions."""
    
    def test_clean_data_removes_customer_id(self, preprocessor, sample_data):
        """Test that customer_id column is removed."""
        cleaned = preprocessor.clean_data(sample_data)
        assert 'customer_id' not in cleaned.columns
    
    def test_clean_data_handles_missing_values(self, preprocessor, sample_data):
        """Test that missing values are handled."""
        sample_data.loc[0, 'tenure'] = np.nan
        cleaned = preprocessor.clean_data(sample_data)
        assert cleaned.shape[0] == sample_data.shape[0] - 1
    
    def test_clean_data_removes_duplicates(self, preprocessor, sample_data):
        """Test that duplicate rows are removed."""
        sample_data = pd.concat([sample_data, sample_data.iloc[[0]]], ignore_index=True)
        cleaned = preprocessor.clean_data(sample_data)
        assert cleaned.shape[0] == sample_data.shape[0] - 1
    
    def test_clean_data_ensures_positive_charges(self, preprocessor, sample_data):
        """Test that negative total_charges are set to 0."""
        sample_data.loc[0, 'total_charges'] = -100.0
        cleaned = preprocessor.clean_data(sample_data)
        assert (cleaned['total_charges'] >= 0).all()


class TestFeatureEngineering:
    """Test feature engineering functions."""
    
    def test_engineer_features_creates_avg_monthly_charge(self, preprocessor, sample_data):
        """Test that average monthly charge is calculated."""
        sample_data = preprocessor.clean_data(sample_data)
        engineered = preprocessor.engineer_features(sample_data)
        assert 'avg_monthly_charge' in engineered.columns
    
    def test_avg_monthly_charge_calculation(self, preprocessor, sample_data):
        """Test average monthly charge is calculated correctly."""
        sample_data = preprocessor.clean_data(sample_data)
        engineered = preprocessor.engineer_features(sample_data)
        
        # Check first row: 600 / (12 + 1) ≈ 46.15
        expected = sample_data.loc[0, 'total_charges'] / (sample_data.loc[0, 'tenure'] + 1)
        assert abs(engineered.loc[0, 'avg_monthly_charge'] - expected) < 0.01
    
    def test_engineer_features_creates_tenure_groups(self, preprocessor, sample_data):
        """Test that tenure groups are created."""
        sample_data = preprocessor.clean_data(sample_data)
        engineered = preprocessor.engineer_features(sample_data)
        assert 'tenure_group' in engineered.columns
        assert engineered['tenure_group'].dtype == object
    
    def test_engineer_features_creates_charge_groups(self, preprocessor, sample_data):
        """Test that charge groups are created."""
        sample_data = preprocessor.clean_data(sample_data)
        engineered = preprocessor.engineer_features(sample_data)
        assert 'charge_group' in engineered.columns
    
    def test_engineer_features_creates_service_score(self, preprocessor, sample_data):
        """Test that service score is calculated."""
        sample_data = preprocessor.clean_data(sample_data)
        engineered = preprocessor.engineer_features(sample_data)
        assert 'service_score' in engineered.columns


class TestFeatureEncoding:
    """Test feature encoding functions."""
    
    def test_encode_features_transforms_categoricals(self, preprocessor, sample_data):
        """Test that categorical features are encoded."""
        sample_data = preprocessor.clean_data(sample_data)
        encoded = preprocessor.encode_features(sample_data, fit=True)
        
        # Check that contract is now numeric
        assert pd.api.types.is_numeric_dtype(encoded['contract'])
    
    def test_label_encoders_are_stored(self, preprocessor, sample_data):
        """Test that label encoders are saved during fit."""
        sample_data = preprocessor.clean_data(sample_data)
        preprocessor.encode_features(sample_data, fit=True)
        
        assert len(preprocessor.label_encoders) > 0
        assert 'contract' in preprocessor.label_encoders


class TestFeatureTargetSplit:
    """Test feature and target separation."""
    
    def test_split_features_target(self, preprocessor, sample_data):
        """Test that features and target are split correctly."""
        sample_data = preprocessor.clean_data(sample_data)
        X, y = preprocessor.split_features_target(sample_data)
        
        assert 'churn' not in X.columns
        assert y.name == 'churn'
        assert len(X) == len(y)
    
    def test_feature_names_stored(self, preprocessor, sample_data):
        """Test that feature names are stored."""
        sample_data = preprocessor.clean_data(sample_data)
        X, y = preprocessor.split_features_target(sample_data)
        
        assert preprocessor.feature_names is not None
        assert len(preprocessor.feature_names) == X.shape[1]


class TestFeatureScaling:
    """Test feature scaling."""
    
    def test_scale_features_returns_array(self, preprocessor, sample_data):
        """Test that scaling returns numpy array."""
        sample_data = preprocessor.clean_data(sample_data)
        sample_data = preprocessor.encode_features(sample_data, fit=True)
        X, y = preprocessor.split_features_target(sample_data)
        
        X_scaled = preprocessor.scale_features(X, fit=True)
        assert isinstance(X_scaled, np.ndarray)
    
    def test_scaled_features_have_mean_zero(self, preprocessor, sample_data):
        """Test that scaled features have approximately mean 0."""
        sample_data = preprocessor.clean_data(sample_data)
        sample_data = preprocessor.encode_features(sample_data, fit=True)
        X, y = preprocessor.split_features_target(sample_data)
        
        X_scaled = preprocessor.scale_features(X, fit=True)
        
        # Mean should be close to 0 for each feature
        assert np.abs(X_scaled.mean(axis=0)).max() < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])