import numpy as np
import pandas as pd
import matplotlib                # <-- added
matplotlib.use('Agg')           # <-- set non-GUI backend before pyplot import
import matplotlib.pyplot as plt  # <-- moved after backend set
import seaborn as sns
import joblib
import os
import re
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, classification_report, confusion_matrix,
                           roc_auc_score)

def train_health_classifier(X, y, model_type='random_forest', random_state=42):
    """
    Train a health classification model
    
    Parameters:
    X (DataFrame): Feature matrix
    y (Series): Target variable (health status)
    model_type (str): Type of model to train ('random_forest', 'xgboost', 'logistic')
    random_state (int): Random seed for reproducibility
    
    Returns:
    model: Trained classification model
    """
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == 'xgboost':
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric='mlogloss'
        )
    elif model_type == 'logistic':
        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train the model
    model.fit(X, y)
    
    return model

def evaluate_classifier(model, X_test, y_test, class_names=None):
    """
    Evaluate a classification model (robust handling when some classes absent in test set)
    """
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Determine labels present in y_test
    labels = np.unique(y_test)
    labels_list = labels.tolist()
    
    # Build classification report safely: pass matching labels and, if possible, matching target_names
    try:
        if class_names is not None:
            # If labels are integer indices and class_names covers them, map names for present labels
            if all(isinstance(l, (int, np.integer)) for l in labels):
                max_label = int(labels.max())
                if max_label < len(class_names):
                    target_names = [class_names[int(l)] for l in labels_list]
                    class_report = classification_report(y_test, y_pred, labels=labels_list, target_names=target_names, zero_division=0)
                else:
                    # class_names length doesn't match label range -> only pass labels
                    class_report = classification_report(y_test, y_pred, labels=labels_list, zero_division=0)
            else:
                # non-integer labels (strings), pass labels only to avoid mismatched target_names
                class_report = classification_report(y_test, y_pred, labels=labels_list, zero_division=0)
        else:
            class_report = classification_report(y_test, y_pred, zero_division=0)
    except Exception:
        # Fallback to a safe call
        class_report = classification_report(y_test, y_pred, zero_division=0)
    
    # Confusion matrix for present labels
    try:
        cm = confusion_matrix(y_test, y_pred, labels=labels_list)
    except Exception:
        cm = confusion_matrix(y_test, y_pred)
    
    # ROC AUC
    try:
        if len(np.unique(y_test)) == 2:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        else:
            auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
    except Exception:
        auc = None
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'classification_report': class_report,
        'confusion_matrix': cm,
        'y_pred': y_pred
    }

def plot_feature_importance(model, feature_names, model_name, asset_type, top_n=15):
    """
    Plot feature importance
    
    Parameters:
    model: Trained model
    feature_names (list): Names of the features
    model_name (str): Name of the model
    asset_type (str): Type of asset
    top_n (int): Number of top features to display
    """
    plt.figure(figsize=(12, 8))
    
    if hasattr(model, 'feature_importances_'):
        # Random Forest and XGBoost
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        
        plt.title(f'Top {top_n} Feature Importances - {model_name} ({asset_type})')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
    elif model_name == 'Logistic Regression':
        # Logistic Regression (coefficients)
        coeffs = model.coef_[0]
        indices = np.argsort(np.abs(coeffs))[-top_n:]
        
        plt.title(f'Top {top_n} Feature Importances - {model_name} ({asset_type})')
        plt.barh(range(len(indices)), coeffs[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Coefficient Value')
    
    plt.tight_layout()
    
    # Save the plot
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(f'results/feature_importance_{model_name.lower().replace(" ", "_")}_{asset_type}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, class_names, model_name, asset_type):
    """
    Plot confusion matrix
    
    Parameters:
    cm (array): Confusion matrix
    class_names (list): Names of the classes
    model_name (str): Name of the model
    asset_type (str): Type of asset
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name} ({asset_type})')
    plt.tight_layout()
    
    # Save the plot
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(f'results/confusion_matrix_{model_name.lower().replace(" ", "_")}_{asset_type}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def tune_random_forest(X_train, y_train, asset_type):
    """
    Perform hyperparameter tuning for Random Forest classifier
    Uses joblib threading backend on Windows to avoid _posixsubprocess errors.
    """
    print(f"\n🔍 Performing hyperparameter tuning for {asset_type}...")
    
    # Define the parameter distribution (remove deprecated 'auto' for max_features)
    param_dist = {
        'n_estimators': sp_randint(100, 300),
        'max_depth': sp_randint(5, 20),
        'min_samples_split': sp_randint(2, 10),
        'min_samples_leaf': sp_randint(1, 5),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    
    # Initialize the classifier
    rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    
    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring='f1_weighted',
        random_state=42,
        n_jobs=-1,   # keep -1 but run under threading backend
        verbose=1
    )
    
    # Fit the random search using a threading backend to avoid forking/posix subprocess on Windows
    try:
        from joblib import parallel_backend
        with parallel_backend('threading'):
            random_search.fit(X_train, y_train)
    except Exception:
        # fallback to single-process fit if anything unexpected happens
        random_search.n_jobs = 1
        random_search.fit(X_train, y_train)
    
    # Get the best estimator
    best_rf = random_search.best_estimator_
    
    # Save the best model
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(best_rf, f'models/best_random_forest_{asset_type}.pkl')
    
    # Save tuning results visualization
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(random_search.cv_results_['mean_test_score'])), 
             random_search.cv_results_['mean_test_score'], 'o-')
    plt.title(f'RandomizedSearchCV Results - {asset_type.capitalize()}')
    plt.xlabel('Iteration')
    plt.ylabel('F1 Score (Weighted)')
    plt.grid(True)
    
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(f'results/hyperparameter_tuning_{asset_type}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print best parameters
    print(f"\nBest parameters for {asset_type}:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"Best F1 Score: {random_search.best_score_:.4f}")
    
    return best_rf, random_search.best_params_, random_search.best_score_

def _format_classification_report(report_str, accuracy=None, precision_na_label='N/A'):
    """
    Adjust sklearn classification_report string:
      - keep class lines as-is
      - replace accuracy row so precision column shows precision_na_label and
        recall & f1 columns contain the numeric accuracy value (if provided)
    """
    lines = report_str.splitlines()
    out_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out_lines.append(line)
            continue

        # accuracy line: replace precision with N/A and put accuracy under recall & f1
        if stripped.startswith('accuracy'):
            # extract support from original line (last token if numeric)
            m = re.search(r'(\d+)\s*$', line)
            support = m.group(1) if m else ''
            if accuracy is None:
                out_lines.append(line)
            else:
                acc_str = f"{accuracy:.2f}"
                out_lines.append(f"{'accuracy':>12} {precision_na_label:>10} {acc_str:>8} {acc_str:>9} {support:>8}")
            continue

        out_lines.append(line)
    return "\n".join(out_lines)

def train_and_evaluate_for_asset(asset_type, data_path, model_type='random_forest'):
    """
    End-to-end training and evaluation for a specific asset type
    
    Parameters:
    asset_type (str): Type of asset ('transformer', 'motor', 'capacitor', 'ups')
    data_path (str): Path to the data file
    model_type (str): Type of model to train
    
    Returns:
    results: Dictionary of evaluation results
    """
    # Import the appropriate preprocessing function
    from data_preprocessing import (
        load_and_preprocess_transformer_data,
        load_and_preprocess_motor_data,
        load_and_preprocess_capacitor_data,
        load_and_preprocess_ups_data
    )
    
    # Load and preprocess data based on asset type
    if asset_type == 'transformer':
        X, y, health_mapping = load_and_preprocess_transformer_data(data_path)
    elif asset_type == 'motor':
        X, y, health_mapping = load_and_preprocess_motor_data(data_path)
    elif asset_type == 'capacitor':
        X, y, health_mapping = load_and_preprocess_capacitor_data(data_path)
    elif asset_type == 'ups':
        X, y, health_mapping = load_and_preprocess_ups_data(data_path)
    else:
        raise ValueError(f"Unknown asset type: {asset_type}")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Balance training set for transformer using SMOTE to improve minority-class learning
    if asset_type == 'transformer':
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=42)
            X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
            X_train, y_train = X_train_res, y_train_res
            # print new distribution
            unique, counts = np.unique(y_train, return_counts=True)
            print(f"Applied SMOTE to transformer training set: {dict(zip(unique, counts))}")
        except Exception:
            print("imblearn.SMOTE not available — continuing without oversampling")
    
    # Hyperparameter tuning for Random Forest
    if model_type == 'random_forest':
        model, best_params, best_score = tune_random_forest(X_train, y_train, asset_type)
    else:
        # Train model
        print(f"📊 Training {model_type} classifier for {asset_type}...")
        model = train_health_classifier(X_train, y_train, model_type=model_type)
    
    # Evaluate model
    class_names = list(health_mapping.keys())
    results = evaluate_classifier(model, X_test, y_test, class_names=class_names)

    # Save model
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(model, f'models/{model_type}_classifier_{asset_type}.pkl')
    print(f"✅ Model saved to models/{model_type}_classifier_{asset_type}.pkl")

    # Plot feature importance
    plot_feature_importance(model, X.columns.tolist(), model_type.capitalize(), asset_type)

    # Plot confusion matrix - ensure labels/names match those present in the test set
    labels_present = np.unique(y_test)
    # build reverse mapping index->name from health_mapping if possible
    rev_map = {v: k for k, v in health_mapping.items()} if health_mapping else {}
    class_names_present = [rev_map.get(int(lbl), str(int(lbl))) for lbl in labels_present]
    try:
        plot_confusion_matrix(results['confusion_matrix'], class_names_present, model_type.capitalize(), asset_type)
    except Exception:
        # fallback to plotting with generic labels
        plot_confusion_matrix(results['confusion_matrix'], [str(l) for l in labels_present], model_type.capitalize(), asset_type)

    # -------------------------
    # Save additional confusion matrix image for the hyperparameter-tuned model
    # (creates a clearly-named file showing tuned model performance)
    # -------------------------
    try:
        tuned_model_name = f"{model_type.capitalize()}_Tuned"
        # use nicer class names if available, else fallback to numeric labels
        if class_names_present:
            plot_confusion_matrix(results['confusion_matrix'], class_names_present, tuned_model_name, asset_type)
        else:
            plot_confusion_matrix(results['confusion_matrix'], [str(l) for l in labels_present], tuned_model_name, asset_type)
        print(f"Saved tuned confusion matrix for {asset_type} as '{tuned_model_name}'")
    except Exception:
        # non-fatal - don't block the pipeline if plotting fails
        print("Could not save tuned confusion matrix image.")

    return results, model, X_test, y_test

if __name__ == "__main__":
    # Example usage
    asset_types = ['transformer', 'motor', 'capacitor', 'ups']

    for asset_type in asset_types:
        print(f"\n{'='*50}")
        print(f"Processing {asset_type} data")
        print(f"{'='*50}")

        # Train and evaluate Random Forest
        results, model, X_test, y_test = train_and_evaluate_for_asset(
            asset_type, f'data/{asset_type}_data.csv', model_type='random_forest'
        )

        print(f"\nRandom Forest Results for {asset_type}:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        if results['auc'] is not None:
            print(f"AUC: {results['auc']:.4f}")

        print("\nClassification Report:")
        # Format classification report so accuracy row shows N/A under precision and accuracy value under recall/f1
        formatted = _format_classification_report(results['classification_report'], accuracy=results.get('accuracy'))
        print(formatted)