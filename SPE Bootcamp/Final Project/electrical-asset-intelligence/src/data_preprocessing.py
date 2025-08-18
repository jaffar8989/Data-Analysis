import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

def load_and_preprocess_transformer_data(file_path):
    """Load and preprocess transformer data"""
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # Create time-based features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Calculate rolling features
    df['oil_temp_7d_avg'] = df['oil_temp_c'].rolling(window=7, min_periods=1).mean()
    df['winding_temp_7d_avg'] = df['winding_temp_c'].rolling(window=7, min_periods=1).mean()
    df['temp_diff'] = df['winding_temp_c'] - df['oil_temp_c']
    df['temp_diff_7d_avg'] = df['temp_diff'].rolling(window=7, min_periods=1).mean()
    
    # Calculate rate of change
    df['oil_temp_rate'] = df['oil_temp_c'].diff().fillna(0)
    df['winding_temp_rate'] = df['winding_temp_c'].diff().fillna(0)
    
    # Drop the date column for modeling
    df = df.drop('date', axis=1)
    
    # Convert categorical features
    df = pd.get_dummies(df, columns=['oil_quality'], prefix='oil_quality')
    
    # Map health status to numeric values
    health_mapping = {'Healthy': 0, 'Warning': 1, 'Critical': 2}
    df['health_status_num'] = df['health_status'].map(health_mapping)
    
    # Define features and target
    X = df.drop(['health_status', 'health_status_num', 'failure_mode'], axis=1)
    y = df['health_status_num']
    
    return X, y, health_mapping

def load_and_preprocess_motor_data(file_path):
    """Load and preprocess motor data"""
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # Create time-based features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Calculate rolling features
    df['vibration_7d_avg'] = df['vibration_mm_s'].rolling(window=7, min_periods=1).mean()
    df['temp_7d_avg'] = df['temperature_c'].rolling(window=7, min_periods=1).mean()
    
    # Calculate rate of change
    df['vibration_rate'] = df['vibration_mm_s'].diff().fillna(0)
    df['temp_rate'] = df['temperature_c'].diff().fillna(0)
    
    # Drop the date column for modeling
    df = df.drop('date', axis=1)
    
    # Convert categorical features
    df = pd.get_dummies(df, columns=['bearing_condition'], prefix='bearing')
    
    # Map health status to numeric values
    health_mapping = {'Healthy': 0, 'Warning': 1, 'Critical': 2}
    df['health_status_num'] = df['health_status'].map(health_mapping)
    
    # Define features and target
    X = df.drop(['health_status', 'health_status_num', 'failure_mode'], axis=1)
    y = df['health_status_num']
    
    return X, y, health_mapping

def load_and_preprocess_capacitor_data(file_path):
    """Load and preprocess capacitor bank data"""
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # Create time-based features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Calculate rolling features
    df['temp_7d_avg'] = df['temperature_c'].rolling(window=7, min_periods=1).mean()
    df['harmonic_7d_avg'] = df['harmonic_distortion_percent'].rolling(window=7, min_periods=1).mean()
    
    # Calculate rate of change
    df['temp_rate'] = df['temperature_c'].diff().fillna(0)
    df['harmonic_rate'] = df['harmonic_distortion_percent'].diff().fillna(0)
    
    # Drop the date column for modeling
    df = df.drop('date', axis=1)
    
    # Convert categorical features
    df = pd.get_dummies(df, columns=['fuse_status'], prefix='fuse')
    
    # Map health status to numeric values
    health_mapping = {'Healthy': 0, 'Warning': 1, 'Critical': 2}
    df['health_status_num'] = df['health_status'].map(health_mapping)
    
    # Define features and target
    X = df.drop(['health_status', 'health_status_num', 'failure_mode'], axis=1)
    y = df['health_status_num']
    
    return X, y, health_mapping

def load_and_preprocess_ups_data(file_path):
    """Load and preprocess UPS data"""
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # Create time-based features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Calculate rolling features
    df['voltage_7d_avg'] = df['voltage_per_cell_v'].rolling(window=7, min_periods=1).mean()
    df['temp_7d_avg'] = df['temperature_c'].rolling(window=7, min_periods=1).mean()
    
    # Calculate rate of change
    df['voltage_rate'] = df['voltage_per_cell_v'].diff().fillna(0)
    df['temp_rate'] = df['temperature_c'].diff().fillna(0)
    
    # Drop the date column for modeling
    df = df.drop('date', axis=1)
    
    # Convert categorical features
    df = pd.get_dummies(df, columns=['operating_mode'], prefix='mode')
    
    # Map health status to numeric values
    health_mapping = {'Healthy': 0, 'Warning': 1, 'Critical': 2}
    df['health_status_num'] = df['health_status'].map(health_mapping)
    
    # Define features and target
    X = df.drop(['health_status', 'health_status_num', 'failure_mode'], axis=1)
    y = df['health_status_num']
    
    return X, y, health_mapping

def create_preprocessing_pipeline(numeric_features, categorical_features):
    """Create a preprocessing pipeline for the data"""
    # Define transformers for numeric and categorical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def explore_data(df, asset_type):
    """Perform exploratory data analysis and save visualizations"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 1. Distribution of health status
    plt.figure(figsize=(10, 6))
    sns.countplot(x='health_status', data=df)
    plt.title(f'{asset_type.capitalize()} Health Status Distribution')
    plt.savefig(f'results/{asset_type}_health_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    plt.figure(figsize=(16, 12))
    sns.heatmap(df[numeric_cols].corr(), annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title(f'{asset_type.capitalize()} Feature Correlation')
    plt.savefig(f'results/{asset_type}_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Time series of key features
    if 'date' in df.columns:
        plt.figure(figsize=(14, 10))
        
        # Oil/Temp for transformers
        if asset_type == 'transformer':
            plt.subplot(2, 1, 1)
            plt.plot(df['date'], df['oil_temp_c'], label='Oil Temp')
            plt.plot(df['date'], df['winding_temp_c'], label='Winding Temp')
            plt.title('Transformer Temperatures Over Time')
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.plot(df['date'], df['insulation_resistance_mohm'], label='Insulation Resistance')
            plt.title('Transformer Insulation Resistance Over Time')
            plt.legend()
        
        # Vibration/Temp for motors
        elif asset_type == 'motor':
            plt.subplot(2, 1, 1)
            plt.plot(df['date'], df['vibration_mm_s'], label='Vibration')
            plt.title('Motor Vibration Over Time')
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.plot(df['date'], df['temperature_c'], label='Temperature')
            plt.title('Motor Temperature Over Time')
            plt.legend()
        
        # Harmonics/Temp for capacitors
        elif asset_type == 'capacitor':
            plt.subplot(2, 1, 1)
            plt.plot(df['date'], df['harmonic_distortion_percent'], label='Harmonic Distortion')
            plt.title('Capacitor Harmonic Distortion Over Time')
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.plot(df['date'], df['temperature_c'], label='Temperature')
            plt.title('Capacitor Temperature Over Time')
            plt.legend()
        
        # Voltage/Temp for UPS
        elif asset_type == 'ups':
            plt.subplot(2, 1, 1)
            plt.plot(df['date'], df['voltage_per_cell_v'], label='Voltage per Cell')
            plt.title('UPS Voltage Over Time')
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.plot(df['date'], df['temperature_c'], label='Temperature')
            plt.title('UPS Temperature Over Time')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'results/{asset_type}_time_series.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ EDA completed for {asset_type}")

if __name__ == "__main__":
    # Example usage for data exploration
    asset_types = ['transformer', 'motor', 'capacitor', 'ups']
    
    for asset_type in asset_types:
        print(f"\nExploring {asset_type} data...")
        df = pd.read_csv(f'data/{asset_type}_data.csv')
        explore_data(df, asset_type)
    
    print("\nAll EDA completed! Check the 'results' directory for visualizations.")