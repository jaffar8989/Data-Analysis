import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
import sqlite3
import warnings
warnings.filterwarnings('ignore')

class TransactionAnomalyDetector:
    def __init__(self, contamination=0.1):
        """
        Initialize the anomaly detector
        
        Args:
            contamination: Expected proportion of outliers (default: 0.1 = 10%)
        """
        self.contamination = contamination
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def generate_sample_data(self, n_samples=10000):
        """
        Generate realistic transaction data with some anomalies
        """
        np.random.seed(42)
        
        # Generate normal transactions
        normal_transactions = []
        
        # Regular transactions (loans, payments, transfers)
        for i in range(int(n_samples * 0.95)):  # 95% normal transactions
            transaction = {
                'transaction_id': f'TXN_{i+1:06d}',
                'user_id': f'USER_{np.random.randint(1, 1000):04d}',
                'transaction_type': np.random.choice(['loan', 'payment', 'transfer'], p=[0.3, 0.5, 0.2]),
                'amount': np.random.lognormal(mean=6, sigma=1),  # Log-normal distribution for amounts
                'timestamp': datetime.now() - timedelta(
                    days=np.random.randint(0, 365),
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60)
                ),
                'account_age_days': np.random.normal(300, 100),
                'previous_transactions': np.random.poisson(20),
                'time_since_last_transaction': np.random.exponential(24),  # hours
                'location_risk_score': np.random.beta(2, 5),  # Most locations are low risk
                'device_trust_score': np.random.beta(8, 2),  # Most devices are trusted
            }
            normal_transactions.append(transaction)
        
        # Generate anomalous transactions (5%)
        anomalous_transactions = []
        for i in range(int(n_samples * 0.05)):
            # Create different types of anomalies
            anomaly_type = np.random.choice(['large_amount', 'suspicious_timing', 'new_account', 'high_risk_location'])
            
            if anomaly_type == 'large_amount':
                transaction = {
                    'transaction_id': f'ANOM_{i+1:06d}',
                    'user_id': f'USER_{np.random.randint(1, 1000):04d}',
                    'transaction_type': np.random.choice(['loan', 'payment', 'transfer']),
                    'amount': np.random.uniform(50000, 500000),  # Unusually large amounts
                    'timestamp': datetime.now() - timedelta(
                        days=np.random.randint(0, 365),
                        hours=np.random.randint(0, 24),
                        minutes=np.random.randint(0, 60)
                    ),
                    'account_age_days': np.random.normal(300, 100),
                    'previous_transactions': np.random.poisson(20),
                    'time_since_last_transaction': np.random.exponential(24),
                    'location_risk_score': np.random.beta(2, 5),
                    'device_trust_score': np.random.beta(8, 2),
                }
            elif anomaly_type == 'suspicious_timing':
                transaction = {
                    'transaction_id': f'ANOM_{i+1:06d}',
                    'user_id': f'USER_{np.random.randint(1, 1000):04d}',
                    'transaction_type': np.random.choice(['loan', 'payment', 'transfer']),
                    'amount': np.random.lognormal(mean=6, sigma=1),
                    'timestamp': datetime.now() - timedelta(
                        days=np.random.randint(0, 365),
                        hours=np.random.randint(2, 5),  # Unusual hours (2-5 AM)
                        minutes=np.random.randint(0, 60)
                    ),
                    'account_age_days': np.random.normal(300, 100),
                    'previous_transactions': np.random.poisson(20),
                    'time_since_last_transaction': np.random.uniform(0, 1),  # Very quick succession
                    'location_risk_score': np.random.beta(2, 5),
                    'device_trust_score': np.random.beta(8, 2),
                }
            elif anomaly_type == 'new_account':
                transaction = {
                    'transaction_id': f'ANOM_{i+1:06d}',
                    'user_id': f'USER_{np.random.randint(1, 1000):04d}',
                    'transaction_type': np.random.choice(['loan', 'payment', 'transfer']),
                    'amount': np.random.lognormal(mean=7, sigma=1),  # Larger amounts
                    'timestamp': datetime.now() - timedelta(
                        days=np.random.randint(0, 365),
                        hours=np.random.randint(0, 24),
                        minutes=np.random.randint(0, 60)
                    ),
                    'account_age_days': np.random.uniform(0, 30),  # Very new accounts
                    'previous_transactions': np.random.poisson(1),  # Few previous transactions
                    'time_since_last_transaction': np.random.exponential(24),
                    'location_risk_score': np.random.beta(2, 5),
                    'device_trust_score': np.random.beta(8, 2),
                }
            else:  # high_risk_location
                transaction = {
                    'transaction_id': f'ANOM_{i+1:06d}',
                    'user_id': f'USER_{np.random.randint(1, 1000):04d}',
                    'transaction_type': np.random.choice(['loan', 'payment', 'transfer']),
                    'amount': np.random.lognormal(mean=6, sigma=1),
                    'timestamp': datetime.now() - timedelta(
                        days=np.random.randint(0, 365),
                        hours=np.random.randint(0, 24),
                        minutes=np.random.randint(0, 60)
                    ),
                    'account_age_days': np.random.normal(300, 100),
                    'previous_transactions': np.random.poisson(20),
                    'time_since_last_transaction': np.random.exponential(24),
                    'location_risk_score': np.random.beta(8, 2),  # High risk location
                    'device_trust_score': np.random.beta(2, 5),  # Untrusted device
                }
            
            anomalous_transactions.append(transaction)
        
        # Combine all transactions
        all_transactions = normal_transactions + anomalous_transactions
        df = pd.DataFrame(all_transactions)
        
        # Add derived features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['amount_log'] = np.log1p(df['amount'])
        
        return df.sample(frac=1).reset_index(drop=True)  # Shuffle the data
    
    def prepare_features(self, df):
        """
        Prepare features for anomaly detection
        """
        # Select numerical features for anomaly detection
        feature_columns = [
            'amount', 'amount_log', 'account_age_days', 'previous_transactions',
            'time_since_last_transaction', 'location_risk_score', 'device_trust_score',
            'hour', 'day_of_week', 'is_weekend'
        ]
        
        # Create transaction type dummy variables
        transaction_dummies = pd.get_dummies(df['transaction_type'], prefix='type')
        
        # Combine features
        features = pd.concat([df[feature_columns], transaction_dummies], axis=1)
        
        # Handle any missing values
        features = features.fillna(features.mean())
        
        return features
    
    def fit_detect(self, df):
        """
        Fit the anomaly detector and predict anomalies
        """
        # Prepare features
        features = self.prepare_features(df)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit isolation forest and predict
        predictions = self.isolation_forest.fit_predict(features_scaled)
        anomaly_scores = self.isolation_forest.decision_function(features_scaled)
        
        # Add results to dataframe
        df['is_anomaly'] = (predictions == -1).astype(int)
        df['anomaly_score'] = anomaly_scores
        
        self.is_fitted = True
        self.feature_names = features.columns.tolist()
        
        return df
    
    def analyze_anomalies(self, df):
        """
        Analyze detected anomalies
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        anomalies = df[df['is_anomaly'] == 1]
        normal_transactions = df[df['is_anomaly'] == 0]
        
        print("=== ANOMALY DETECTION RESULTS ===")
        print(f"Total transactions: {len(df)}")
        print(f"Anomalies detected: {len(anomalies)} ({len(anomalies)/len(df)*100:.2f}%)")
        print()
        
        # Analyze anomaly patterns
        print("=== ANOMALY ANALYSIS ===")
        print("\nTransaction Type Distribution:")
        print("Normal transactions:")
        print(normal_transactions['transaction_type'].value_counts(normalize=True))
        print("\nAnomalous transactions:")
        print(anomalies['transaction_type'].value_counts(normalize=True))
        
        print(f"\nAmount Statistics:")
        print(f"Normal transactions - Mean: ${normal_transactions['amount'].mean():.2f}, Median: ${normal_transactions['amount'].median():.2f}")
        print(f"Anomalous transactions - Mean: ${anomalies['amount'].mean():.2f}, Median: ${anomalies['amount'].median():.2f}")
        
        print(f"\nAccount Age (days):")
        print(f"Normal transactions - Mean: {normal_transactions['account_age_days'].mean():.1f}")
        print(f"Anomalous transactions - Mean: {anomalies['account_age_days'].mean():.1f}")
        
        print(f"\nLocation Risk Score:")
        print(f"Normal transactions - Mean: {normal_transactions['location_risk_score'].mean():.3f}")
        print(f"Anomalous transactions - Mean: {anomalies['location_risk_score'].mean():.3f}")
        
        return anomalies
    
    def visualize_anomalies(self, df):
        """
        Create visualizations of anomalies
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Transaction Anomaly Detection Analysis', fontsize=16, fontweight='bold')
        
        # 1. Amount distribution
        axes[0, 0].hist(df[df['is_anomaly'] == 0]['amount'], bins=50, alpha=0.7, 
                       label='Normal', color='green', density=True)
        axes[0, 0].hist(df[df['is_anomaly'] == 1]['amount'], bins=50, alpha=0.7, 
                       label='Anomaly', color='red', density=True)
        axes[0, 0].set_xlabel('Transaction Amount')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Amount Distribution')
        axes[0, 0].legend()
        axes[0, 0].set_xlim(0, 50000)  # Focus on main distribution
        
        # 2. Amount vs Account Age scatter
        normal = df[df['is_anomaly'] == 0]
        anomaly = df[df['is_anomaly'] == 1]
        
        axes[0, 1].scatter(normal['account_age_days'], normal['amount'], 
                          alpha=0.5, s=20, color='green', label='Normal')
        axes[0, 1].scatter(anomaly['account_age_days'], anomaly['amount'], 
                          alpha=0.8, s=30, color='red', label='Anomaly')
        axes[0, 1].set_xlabel('Account Age (days)')
        axes[0, 1].set_ylabel('Transaction Amount')
        axes[0, 1].set_title('Amount vs Account Age')
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 50000)
        
        # 3. Time patterns
        hour_anomalies = df[df['is_anomaly'] == 1]['hour'].value_counts().sort_index()
        hour_normal = df[df['is_anomaly'] == 0]['hour'].value_counts().sort_index()
        
        axes[0, 2].bar(hour_normal.index, hour_normal.values, alpha=0.7, 
                      label='Normal', color='green')
        axes[0, 2].bar(hour_anomalies.index, hour_anomalies.values, alpha=0.7, 
                      label='Anomaly', color='red')
        axes[0, 2].set_xlabel('Hour of Day')
        axes[0, 2].set_ylabel('Number of Transactions')
        axes[0, 2].set_title('Hourly Transaction Pattern')
        axes[0, 2].legend()
        
        # 4. Risk scores
        axes[1, 0].scatter(normal['location_risk_score'], normal['device_trust_score'], 
                          alpha=0.5, s=20, color='green', label='Normal')
        axes[1, 0].scatter(anomaly['location_risk_score'], anomaly['device_trust_score'], 
                          alpha=0.8, s=30, color='red', label='Anomaly')
        axes[1, 0].set_xlabel('Location Risk Score')
        axes[1, 0].set_ylabel('Device Trust Score')
        axes[1, 0].set_title('Risk Score Distribution')
        axes[1, 0].legend()
        
        # 5. Anomaly score distribution
        axes[1, 1].hist(df[df['is_anomaly'] == 0]['anomaly_score'], bins=50, alpha=0.7, 
                       label='Normal', color='green', density=True)
        axes[1, 1].hist(df[df['is_anomaly'] == 1]['anomaly_score'], bins=50, alpha=0.7, 
                       label='Anomaly', color='red', density=True)
        axes[1, 1].set_xlabel('Anomaly Score')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Anomaly Score Distribution')
        axes[1, 1].legend()
        axes[1, 1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # 6. Transaction type analysis
        type_counts = df.groupby(['transaction_type', 'is_anomaly']).size().unstack(fill_value=0)
        type_counts.plot(kind='bar', ax=axes[1, 2], color=['green', 'red'])
        axes[1, 2].set_xlabel('Transaction Type')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Anomalies by Transaction Type')
        axes[1, 2].legend(['Normal', 'Anomaly'])
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def generate_fraud_prevention_report(self, df):
        """
        Generate a comprehensive fraud prevention report
        """
        anomalies = df[df['is_anomaly'] == 1].copy()
        
        # Priority scoring based on multiple factors
        anomalies['priority_score'] = (
            (anomalies['amount'] > anomalies['amount'].quantile(0.95)) * 0.3 +
            (anomalies['account_age_days'] < 30) * 0.2 +
            (anomalies['location_risk_score'] > 0.7) * 0.2 +
            (anomalies['device_trust_score'] < 0.3) * 0.2 +
            (anomalies['time_since_last_transaction'] < 1) * 0.1
        )
        
        # Categorize anomalies
        def categorize_anomaly(row):
            if row['amount'] > df['amount'].quantile(0.95):
                return 'Large Amount'
            elif row['account_age_days'] < 30:
                return 'New Account'
            elif row['location_risk_score'] > 0.7:
                return 'High Risk Location'
            elif row['time_since_last_transaction'] < 1:
                return 'Rapid Succession'
            else:
                return 'Pattern Anomaly'
        
        anomalies['anomaly_category'] = anomalies.apply(categorize_anomaly, axis=1)
        
        # Generate report
        report = []
        report.append("=" * 60)
        report.append("FRAUD PREVENTION SYSTEM - ANOMALY REPORT")
        report.append("=" * 60)
        report.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Transactions Analyzed: {len(df):,}")
        report.append(f"Anomalies Detected: {len(anomalies):,}")
        report.append(f"Anomaly Rate: {len(anomalies)/len(df)*100:.2f}%")
        report.append("")
        
        # High priority anomalies
        high_priority = anomalies[anomalies['priority_score'] >= 0.5].sort_values('priority_score', ascending=False)
        report.append(f"HIGH PRIORITY ALERTS: {len(high_priority)}")
        report.append("-" * 40)
        
        for _, row in high_priority.head(10).iterrows():
            report.append(f"Transaction ID: {row['transaction_id']}")
            report.append(f"  User: {row['user_id']}")
            report.append(f"  Amount: ${row['amount']:,.2f}")
            report.append(f"  Type: {row['transaction_type']}")
            report.append(f"  Category: {row['anomaly_category']}")
            report.append(f"  Priority Score: {row['priority_score']:.2f}")
            report.append(f"  Anomaly Score: {row['anomaly_score']:.3f}")
            report.append("")
        
        # Category breakdown
        report.append("ANOMALY CATEGORIES:")
        report.append("-" * 40)
        category_counts = anomalies['anomaly_category'].value_counts()
        for category, count in category_counts.items():
            percentage = count / len(anomalies) * 100
            report.append(f"{category}: {count} ({percentage:.1f}%)")
        
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 40)
        report.append("1. Implement real-time monitoring for large transactions (>$50,000)")
        report.append("2. Add additional verification for new accounts (<30 days)")
        report.append("3. Review transactions from high-risk locations")
        report.append("4. Set up alerts for rapid transaction succession")
        report.append("5. Consider implementing dynamic risk scoring")
        report.append("6. Regular model retraining with new data")
        
        return "\n".join(report)
    
    def save_to_database(self, df, db_path='transactions.db'):
        """
        Save transactions and anomaly results to SQLite database
        """
        conn = sqlite3.connect(db_path)
        
        # Create tables
        cursor = conn.cursor()
        
        # Transactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                user_id TEXT,
                transaction_type TEXT,
                amount REAL,
                timestamp DATETIME,
                account_age_days INTEGER,
                previous_transactions INTEGER,
                time_since_last_transaction REAL,
                location_risk_score REAL,
                device_trust_score REAL,
                hour INTEGER,
                day_of_week INTEGER,
                is_weekend INTEGER,
                is_anomaly INTEGER,
                anomaly_score REAL
            )
        ''')
        
        # Save data
        df.to_sql('transactions', conn, if_exists='replace', index=False)
        
        # Create anomalies view
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS anomalies AS
            SELECT * FROM transactions WHERE is_anomaly = 1
            ORDER BY anomaly_score ASC
        ''')
        
        conn.commit()
        conn.close()
        
        print(f"Data saved to {db_path}")
        print("Tables created: transactions")
        print("Views created: anomalies")

# Example usage and demonstration
def main():
    # Initialize the anomaly detector
    detector = TransactionAnomalyDetector(contamination=0.05)
    
    # Generate sample data
    print("Generating sample transaction data...")
    df = detector.generate_sample_data(n_samples=5000)
    
    # Fit model and detect anomalies
    print("Training anomaly detection model...")
    df = detector.fit_detect(df)
    
    # Analyze results
    anomalies = detector.analyze_anomalies(df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    detector.visualize_anomalies(df)
    
    # Generate fraud prevention report
    print("\nGenerating fraud prevention report...")
    report = detector.generate_fraud_prevention_report(df)
    print(report)
    
    # Save to database
    print("\nSaving results to database...")
    detector.save_to_database(df)
    
    # Show top anomalies
    print("\n=== TOP 10 ANOMALIES ===")
    top_anomalies = df[df['is_anomaly'] == 1].nsmallest(10, 'anomaly_score')
    for _, row in top_anomalies.iterrows():
        print(f"ID: {row['transaction_id']}, Amount: ${row['amount']:,.2f}, "
              f"Type: {row['transaction_type']}, Score: {row['anomaly_score']:.3f}")

if __name__ == "__main__":
    main()