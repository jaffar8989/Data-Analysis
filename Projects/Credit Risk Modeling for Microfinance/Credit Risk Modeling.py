import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           precision_score, recall_score, f1_score)
from sklearn.ensemble import RandomForestClassifier
import sqlite3
import os

# Set random seed for reproducibility
np.random.seed(42)

class MicrofinanceCreditRiskModel:
    """
    Credit Risk Modeling System for Microfinance with Shariah-Compliant Features
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.label_encoders = {}
        self.X_test = None
        self.y_test = None
        
    def generate_synthetic_data(self, n_samples=5000):
        """
        Generate synthetic microfinance data with realistic features
        """
        print("Generating synthetic microfinance dataset...")
        
        try:
            # Set seed for reproducibility
            np.random.seed(42)
            
            # Basic demographics
            ages = np.random.normal(35, 12, n_samples)
            ages = np.clip(ages, 18, 75).astype(int)
            
            # Gender distribution
            genders = np.random.choice(['Male', 'Female'], n_samples, p=[0.4, 0.6])
            
            # Education levels
            education_levels = np.random.choice(['Primary', 'Secondary', 'Higher'], 
                                              n_samples, p=[0.4, 0.45, 0.15])
            
            # Business types (Shariah-compliant)
            business_types = np.random.choice([
                'Agriculture', 'Small_Trade', 'Handicrafts', 'Services', 
                'Manufacturing', 'Transportation', 'Food_Processing'
            ], n_samples, p=[0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05])
            
            # Monthly income (in local currency)
            base_income = np.random.lognormal(6, 0.8, n_samples)
            monthly_income = np.clip(base_income, 100, 5000).round(2)
            
            # Household size
            household_size = np.random.poisson(4, n_samples) + 1
            household_size = np.clip(household_size, 1, 12)
            
            # Loan amount requested
            loan_multiplier = np.random.uniform(0.5, 5.0, n_samples)
            loan_amount = (loan_multiplier * monthly_income).round(2)
            
            # Loan term (months)
            loan_term = np.random.choice([6, 12, 18, 24, 36], n_samples, 
                                       p=[0.1, 0.3, 0.3, 0.2, 0.1])
            
            # Credit history features
            previous_loans = np.random.poisson(2, n_samples)
            previous_loans = np.clip(previous_loans, 0, 10)
            
            # Payment history (percentage of on-time payments)
            payment_history = np.random.beta(8, 2, n_samples)
            # Adjust for those with no previous loans
            payment_history = np.where(previous_loans == 0, 1.0, payment_history)
            
            # Assets value
            assets_value = np.random.lognormal(7, 1.2, n_samples)
            assets_value = np.clip(assets_value, 50, 20000).round(2)
            
            # Debt-to-income ratio
            existing_debt = np.random.exponential(monthly_income * 0.3, n_samples)
            debt_to_income = existing_debt / monthly_income
            debt_to_income = np.clip(debt_to_income, 0, 2).round(4)
            
            # Employment duration (months)
            employment_duration = np.random.exponential(24, n_samples)
            employment_duration = np.clip(employment_duration, 1, 120).round(0).astype(int)
            
            # Group lending participation
            group_member = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
            
            # Savings account
            has_savings = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
            savings_balance = np.where(has_savings, 
                                     np.random.exponential(monthly_income * 0.5, n_samples), 
                                     0).round(2)
            
            # Location risk (urban vs rural)
            location_type = np.random.choice(['Urban', 'Rural'], n_samples, p=[0.35, 0.65])
            
            # Seasonal business indicator
            seasonal_business = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
            
            # Create target variable (default risk)
            # Higher risk factors: high debt-to-income, low payment history, seasonal business
            risk_score = (
                (debt_to_income * 0.3) + 
                ((1 - payment_history) * 0.25) + 
                (seasonal_business * 0.1) + 
                ((loan_amount / monthly_income) * 0.2) + 
                (np.random.normal(0, 0.15, n_samples))
            )
            
            # Convert to binary default indicator
            default_threshold = np.percentile(risk_score, 80)  # 20% default rate
            default = (risk_score > default_threshold).astype(int)
            
            # Create DataFrame
            data = pd.DataFrame({
                'age': ages,
                'gender': genders,
                'education': education_levels,
                'business_type': business_types,
                'monthly_income': monthly_income,
                'household_size': household_size,
                'loan_amount': loan_amount,
                'loan_term': loan_term,
                'previous_loans': previous_loans,
                'payment_history': payment_history,
                'assets_value': assets_value,
                'debt_to_income': debt_to_income,
                'employment_duration': employment_duration,
                'group_member': group_member,
                'has_savings': has_savings,
                'savings_balance': savings_balance,
                'location_type': location_type,
                'seasonal_business': seasonal_business,
                'default': default
            })
            
            print(f"Successfully generated {len(data)} synthetic records")
            print(f"Default rate: {data['default'].mean():.2%}")
            return data
            
        except Exception as e:
            print(f"Error generating synthetic data: {e}")
            return None
    
    def create_database(self, data):
        """
        Create SQLite database for the dataset
        """
        try:
            # Remove existing database if it exists
            db_path = 'microfinance_data.db'
            if os.path.exists(db_path):
                os.remove(db_path)
            
            conn = sqlite3.connect(db_path)
            data.to_sql('loans', conn, if_exists='replace', index=False)
            
            print("Database created successfully!")
            print(f"Total records: {len(data)}")
            
            # Show sample queries
            print("\nSample SQL Query - Default rate by business type:")
            query1 = """
            SELECT business_type, 
                   COUNT(*) as total_loans,
                   SUM(default) as defaults,
                   ROUND(CAST(SUM(default) AS FLOAT) / COUNT(*) * 100, 2) as default_rate_pct
            FROM loans 
            GROUP BY business_type 
            ORDER BY default_rate_pct DESC;
            """
            result1 = pd.read_sql_query(query1, conn)
            print(result1.to_string(index=False))
            
            conn.close()
            return data
            
        except Exception as e:
            print(f"Error creating database: {e}")
            return data
    
    def feature_engineering(self, data):
        """
        Engineer features for credit risk modeling
        """
        print("Engineering features...")
        
        try:
            # Create a copy to avoid modifying original data
            df = data.copy()
            
            # 1. Loan-to-Income Ratio
            df['loan_to_income'] = df['loan_amount'] / df['monthly_income']
            
            # 2. Monthly Payment (assuming profit-sharing)
            # Using profit-sharing ratio instead of interest for Shariah compliance
            profit_sharing_rate = 0.08  # 8% profit sharing
            df['monthly_payment'] = (df['loan_amount'] * (1 + profit_sharing_rate)) / df['loan_term']
            
            # 3. Payment-to-Income Ratio
            df['payment_to_income'] = df['monthly_payment'] / df['monthly_income']
            
            # 4. Assets-to-Loan Ratio
            df['assets_to_loan'] = df['assets_value'] / df['loan_amount']
            
            # 5. Savings-to-Income Ratio
            df['savings_to_income'] = df['savings_balance'] / df['monthly_income']
            
            # 6. Experience Score (combination of employment duration and previous loans)
            df['experience_score'] = (
                (df['employment_duration'] / 12) * 0.6 + 
                (df['previous_loans'] / 10) * 0.4
            )
            
            # 7. Risk Categories
            high_risk_businesses = ['Agriculture', 'Transportation']
            df['high_risk_business'] = df['business_type'].isin(high_risk_businesses).astype(int)
            df['vulnerable_group'] = ((df['gender'] == 'Female') & (df['household_size'] > 5)).astype(int)
            
            # 8. Stability Score
            df['stability_score'] = (
                (df['payment_history'] * 0.4) + 
                (df['group_member'] * 0.2) + 
                (df['has_savings'] * 0.2) + 
                ((df['employment_duration'] > 12) * 0.2)
            )
            
            # Handle any infinite or NaN values
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(0)
            
            print(f"Feature engineering complete. Total features: {len(df.columns)}")
            return df
            
        except Exception as e:
            print(f"Error in feature engineering: {e}")
            return data
    
    def preprocess_data(self, data):
        """
        Preprocess data for machine learning
        """
        print("Preprocessing data...")
        
        try:
            # Separate features and target
            X = data.drop('default', axis=1)
            y = data['default']
            
            # Handle categorical variables
            categorical_cols = ['gender', 'education', 'business_type', 'location_type']
            
            for col in categorical_cols:
                if col in X.columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.label_encoders[col] = le
            
            # Store feature columns
            self.feature_columns = X.columns.tolist()
            
            # Scale numerical features
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns)
            
            print(f"Data preprocessing complete. Shape: {X_scaled.shape}")
            return X_scaled, y
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None, None
    
    def train_model(self, X, y):
        """
        Train logistic regression model with hyperparameter tuning
        """
        print("Training credit risk model...")
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Store test data for evaluation
            self.X_test = X_test
            self.y_test = y_test
            
            # Hyperparameter tuning
            param_grid = {
                'C': [0.1, 1, 10],
                'class_weight': ['balanced', None],
                'solver': ['liblinear', 'lbfgs']
            }
            
            lr = LogisticRegression(random_state=42, max_iter=1000)
            grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc', 
                                     n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
            return self.model
            
        except Exception as e:
            print(f"Error training model: {e}")
            return None
    
    def evaluate_model(self):
        """
        Comprehensive model evaluation
        """
        if self.model is None:
            print("Model not trained yet!")
            return None
        
        if self.X_test is None or self.y_test is None:
            print("Test data not available!")
            return None
        
        print("Evaluating model performance...")
        
        try:
            # Predictions
            y_pred = self.model.predict(self.X_test)
            y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            print(f"\nModel Performance Metrics:")
            print(f"ROC-AUC Score: {auc_score:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred))
            
            # Feature importance
            if hasattr(self.model, 'coef_'):
                feature_importance = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': abs(self.model.coef_[0])
                }).sort_values('importance', ascending=False)
                
                print("\nTop 10 Most Important Features:")
                print(feature_importance.head(10).to_string(index=False))
            
            return {
                'auc_score': auc_score,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'feature_importance': feature_importance if hasattr(self.model, 'coef_') else None
            }
            
        except Exception as e:
            print(f"Error evaluating model: {e}")
            return None
    
    def plot_results(self):
        """
        Create visualizations for model performance
        """
        if self.model is None or self.X_test is None or self.y_test is None:
            print("Model not trained or test data not available!")
            return
        
        try:
            y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
            y_pred = self.model.predict(self.X_test)
            
            # Set up the plotting style
            plt.style.use('default')
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. ROC Curve
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            
            axes[0, 0].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})', linewidth=2)
            axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            axes[0, 0].set_xlabel('False Positive Rate')
            axes[0, 0].set_ylabel('True Positive Rate')
            axes[0, 0].set_title('ROC Curve')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
            
            axes[0, 1].plot(recall, precision, label='Precision-Recall Curve', linewidth=2)
            axes[0, 1].set_xlabel('Recall')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].set_title('Precision-Recall Curve')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
            axes[1, 0].set_title('Confusion Matrix')
            axes[1, 0].set_ylabel('Actual')
            axes[1, 0].set_xlabel('Predicted')
            
            # 4. Feature Importance
            if hasattr(self.model, 'coef_'):
                feature_importance = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': abs(self.model.coef_[0])
                }).sort_values('importance', ascending=False).head(10)
                
                axes[1, 1].barh(range(len(feature_importance)), 
                               feature_importance['importance'])
                axes[1, 1].set_yticks(range(len(feature_importance)))
                axes[1, 1].set_yticklabels(feature_importance['feature'])
                axes[1, 1].set_xlabel('Importance')
                axes[1, 1].set_title('Top 10 Feature Importance')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    def shariah_compliance_recommendations(self, data):
        """
        Generate Shariah-compliant lending recommendations
        """
        print("\n" + "="*60)
        print("SHARIAH-COMPLIANT MICROFINANCE RECOMMENDATIONS")
        print("="*60)
        
        try:
            # Risk analysis by business type
            risk_analysis = data.groupby('business_type').agg({
                'default': ['count', 'sum', 'mean'],
                'loan_amount': 'mean',
                'monthly_income': 'mean'
            }).round(4)
            
            print("\n1. BUSINESS TYPE RISK ANALYSIS:")
            print("-" * 50)
            print(f"{'Business Type':<15} | {'Default Rate':<12} | {'Risk Level':<10} | {'Total Loans':<12}")
            print("-" * 50)
            
            for business in risk_analysis.index:
                default_rate = risk_analysis.loc[business, ('default', 'mean')]
                total_loans = int(risk_analysis.loc[business, ('default', 'count')])
                
                if default_rate > 0.25:
                    risk_level = "HIGH"
                elif default_rate > 0.15:
                    risk_level = "MEDIUM"
                else:
                    risk_level = "LOW"
                
                print(f"{business:<15} | {default_rate:>10.2%} | {risk_level:<10} | {total_loans:>10}")
            
            print("\n2. SHARIAH-COMPLIANT RECOMMENDATIONS:")
            print("-" * 50)
            
            recommendations = [
                "• Implement Profit-Loss Sharing (Mudarabah) for agriculture loans",
                "• Use Asset-backed financing (Murabaha) for equipment purchases",
                "• Establish Islamic cooperative groups for women entrepreneurs",
                "• Implement Takaful (Islamic insurance) for seasonal businesses",
                "• Create Qard Hassan (benevolent loans) for emergency situations",
                "• Develop Wakala (agency) structures for group lending",
                "• Establish Shariah-compliant collateral alternatives (Rahn)",
                "• Implement Islamic microfinance education programs"
            ]
            
            for rec in recommendations:
                print(rec)
            
            print("\n3. RISK MITIGATION STRATEGIES:")
            print("-" * 50)
            
            # High-risk segments
            high_debt = (data['debt_to_income'] > 0.5).sum()
            poor_payment = (data['payment_history'] < 0.7).sum()
            seasonal_biz = (data['seasonal_business'] == 1).sum()
            high_loan_ratio = (data['loan_amount'] / data['monthly_income'] > 3).sum()
            
            total_clients = len(data)
            
            print(f"• High debt-to-income ratio: {high_debt:4d} clients ({high_debt/total_clients*100:5.1f}%)")
            print(f"• Poor payment history: {poor_payment:4d} clients ({poor_payment/total_clients*100:5.1f}%)")
            print(f"• Seasonal business model: {seasonal_biz:4d} clients ({seasonal_biz/total_clients*100:5.1f}%)")
            print(f"• High loan-to-income ratio: {high_loan_ratio:4d} clients ({high_loan_ratio/total_clients*100:5.1f}%)")
            
            print("\n4. PORTFOLIO OPTIMIZATION:")
            print("-" * 50)
            
            # Portfolio metrics
            total_portfolio = data['loan_amount'].sum()
            avg_loan_size = data['loan_amount'].mean()
            default_rate = data['default'].mean()
            
            print(f"• Total Portfolio Value: ${total_portfolio:,.0f}")
            print(f"• Average Loan Size: ${avg_loan_size:,.0f}")
            print(f"• Overall Default Rate: {default_rate:.2%}")
            print(f"• Recommended Maximum Exposure per Sector: 25%")
            print(f"• Recommended Group Lending Ratio: 70%+")
            
            return {
                'risk_analysis': risk_analysis,
                'portfolio_metrics': {
                    'total_portfolio': total_portfolio,
                    'avg_loan_size': avg_loan_size,
                    'default_rate': default_rate
                }
            }
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return None

def main():
    """
    Main execution function
    """
    print("MICROFINANCE CREDIT RISK MODELING SYSTEM")
    print("="*50)
    
    try:
        # Initialize model
        model = MicrofinanceCreditRiskModel()
        
        # Generate synthetic data
        print("\n1. Generating Data...")
        data = model.generate_synthetic_data(n_samples=5000)
        if data is None:
            print("Failed to generate data. Exiting.")
            return None, None, None
        
        # Create database
        print("\n2. Creating Database...")
        data = model.create_database(data)
        
        # Feature engineering
        print("\n3. Feature Engineering...")
        data_engineered = model.feature_engineering(data)
        
        # Preprocess data
        print("\n4. Preprocessing Data...")
        X, y = model.preprocess_data(data_engineered)
        if X is None or y is None:
            print("Failed to preprocess data. Exiting.")
            return None, None, None
        
        # Train model
        print("\n5. Training Model...")
        trained_model = model.train_model(X, y)
        if trained_model is None:
            print("Failed to train model. Exiting.")
            return None, None, None
        
        # Evaluate model
        print("\n6. Evaluating Model...")
        results = model.evaluate_model()
        
        # Generate recommendations
        print("\n7. Generating Recommendations...")
        recommendations = model.shariah_compliance_recommendations(data)
        
        # Plot results
        print("\n8. Creating Visualizations...")
        model.plot_results()
        
        print("\n" + "="*50)
        print("CREDIT RISK MODELING COMPLETE")
        print("="*50)
        
        return model, data, results
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        return None, None, None

# Run the main function
if __name__ == "__main__":
    model, data, results = main()
    
    # Display summary if successful
    if model is not None and data is not None:
        print("\n📊 FINAL SUMMARY:")
        print(f"✅ Dataset: {len(data)} records generated")
        print(f"✅ Features: {len(model.feature_columns)} engineered")
        print(f"✅ Model: Logistic Regression trained")
        if results:
            print(f"✅ Performance: AUC = {results['auc_score']:.3f}")
        print("✅ Shariah-compliant recommendations generated")
        print("\n🎯 Ready for production deployment!")
    else:
        print("\n❌ Model training failed. Please check the error messages above.")