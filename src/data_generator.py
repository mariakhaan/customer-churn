"""
Data Generator Module

Generates synthetic customer data for churn prediction.
"""

import numpy as np
import pandas as pd
from typing import Tuple


class ChurnDataGenerator:
    """Generate synthetic customer churn data."""
    
    def __init__(self, n_samples: int = 5000, random_state: int = 42):
        """
        Initialize the data generator.
        
        Args:
            n_samples: Number of samples to generate
            random_state: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate(self) -> pd.DataFrame:
        """
        Generate synthetic customer data.
        
        Returns:
            DataFrame with customer features and churn labels
        """
        # Generate features
        tenure = np.random.randint(1, 73, self.n_samples)
        monthly_charges = np.random.uniform(20, 120, self.n_samples)
        total_charges = tenure * monthly_charges + np.random.normal(0, 100, self.n_samples)
        total_charges = np.maximum(total_charges, 0)  # Ensure non-negative
        
        contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                   self.n_samples, p=[0.5, 0.3, 0.2])
        payment_method = np.random.choice(['Electronic check', 'Mailed check', 
                                          'Bank transfer', 'Credit card'], 
                                         self.n_samples)
        internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], 
                                           self.n_samples, p=[0.4, 0.4, 0.2])
        tech_support = np.random.choice(['Yes', 'No', 'No internet service'], 
                                       self.n_samples, p=[0.3, 0.5, 0.2])
        num_services = np.random.randint(0, 7, self.n_samples)
        
        # Generate churn with realistic patterns
        churn_probability = self._calculate_churn_probability(
            tenure, monthly_charges, contract, internet_service
        )
        churn = (np.random.random(self.n_samples) < churn_probability).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame({
            'customer_id': [f'CUST_{i:05d}' for i in range(self.n_samples)],
            'tenure': tenure,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'contract': contract,
            'payment_method': payment_method,
            'internet_service': internet_service,
            'tech_support': tech_support,
            'num_services': num_services,
            'churn': churn
        })
        
        return df
    
    def _calculate_churn_probability(
        self, 
        tenure: np.ndarray, 
        monthly_charges: np.ndarray,
        contract: np.ndarray,
        internet_service: np.ndarray
    ) -> np.ndarray:
        """
        Calculate churn probability based on features.
        
        Realistic patterns:
        - Higher churn for short tenure
        - Higher churn for high monthly charges
        - Higher churn for month-to-month contracts
        - Higher churn for fiber optic (pricing)
        """
        base_prob = 0.1
        
        # Tenure effect (decreases with longer tenure)
        tenure_factor = np.exp(-tenure / 12) * 0.5
        
        # Monthly charges effect (increases with higher charges)
        charge_factor = (monthly_charges - 20) / 100 * 0.2
        
        # Contract effect
        contract_factor = np.where(contract == 'Month-to-month', 0.5,
                          np.where(contract == 'One year', 0.05, 0.0))
        
        # Internet service effect
        internet_factor = np.where(internet_service == 'Fiber optic', 0.15, 0.0)
        
        churn_prob = base_prob + tenure_factor + charge_factor + contract_factor + internet_factor
        
        # Clip between 0 and 1
        return np.clip(churn_prob, 0, 1)
    
    def save(self, df: pd.DataFrame, filepath: str) -> None:
        """
        Save generated data to CSV.
        
        Args:
            df: DataFrame to save
            filepath: Output file path
        """
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        print(f"Shape: {df.shape}")
        print(f"Churn rate: {df['churn'].mean():.2%}")


def main():
    """Generate and save synthetic data."""
    generator = ChurnDataGenerator(n_samples=5000, random_state=42)
    df = generator.generate()
    generator.save(df, "data/customer_data.csv")


if __name__ == "__main__":
    main()
    