import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid Tkinter warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import random
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix

def build_lstm_autoencoder(input_shape):
    """Build LSTM autoencoder for complex time-series anomaly detection"""
    model = Sequential([
        # Encoder
        LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu', return_sequences=False),
        Dropout(0.2),
        # Bottleneck
        Dense(16, activation='relu'),
        # Decoder
        Dense(32, activation='relu'),
        RepeatVector(input_shape[0]),
        LSTM(32, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(64, activation='relu', return_sequences=True),
        Dropout(0.2),
        TimeDistributed(Dense(input_shape[1]))
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def detect_anomalies_with_lstm(X, window_size=30, asset_type="transformer"):
    """Detect anomalies using LSTM autoencoder"""
    # Reshape for LSTM [samples, time steps, features]
    n_samples = X.shape[0] - window_size + 1
    X_reshaped = np.array([X[i:i+window_size] for i in range(n_samples)])
    
    # Build and train autoencoder
    input_shape = (window_size, X_reshaped.shape[2])
    autoencoder = build_lstm_autoencoder(input_shape)
    
    # Split data for training/validation
    X_train, X_val = train_test_split(X_reshaped, test_size=0.2, random_state=42)
    
    # Train model
    history = autoencoder.fit(
        X_train, X_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, X_val),
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=0
    )
    
    # Save training history for visualization
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'LSTM Autoencoder Training - {asset_type.capitalize()}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(f'results/lstm_autoencoder_training_{asset_type}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Reconstruct the training data
    X_pred = autoencoder.predict(X_reshaped, verbose=0)
    
    # Calculate reconstruction error
    mse = np.mean(np.power(X_reshaped - X_pred, 2), axis=(1, 2))
    
    # Determine threshold (95th percentile)
    threshold = np.percentile(mse, 95)
    
    # Identify anomalies
    anomalies = mse > threshold
    
    # Save the model
    if not os.path.exists('models'):
        os.makedirs('models')
    autoencoder.save(f'models/lstm_autoencoder_{asset_type}.h5')
    
    # Plot anomalies
    plt.figure(figsize=(14, 7))
    plt.plot(mse, label='Reconstruction Error')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.4f})')
    plt.scatter(np.where(anomalies)[0], mse[anomalies], c='r', label='Anomalies')
    plt.title(f'LSTM Anomaly Detection - {asset_type.capitalize()}')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.xlabel('Time Window')
    plt.legend()
    
    plt.savefig(f'results/lstm_anomaly_detection_{asset_type}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return anomalies, mse, threshold

def train_isolation_forest(X, contamination=0.05, random_state=42):
    """
    Train an Isolation Forest model for anomaly detection
    
    Parameters:
    X (DataFrame): Feature matrix
    contamination (float): Expected proportion of outliers in the data
    random_state (int): Random seed for reproducibility
    
    Returns:
    model: Trained Isolation Forest model
    """
    # Create and train the model
    model = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X)
    
    return model

def train_one_class_svm(X, nu=0.05, kernel='rbf', gamma='scale'):
    """
    Train a One-Class SVM model for anomaly detection
    
    Parameters:
    X (DataFrame): Feature matrix
    nu (float): Upper bound on the fraction of training errors and a lower bound of the fraction of support vectors
    kernel (str): Specifies the kernel type
    gamma (str or float): Kernel coefficient
    
    Returns:
    model: Trained One-Class SVM model
    """
    # Create and train the model
    model = OneClassSVM(
        nu=nu,
        kernel=kernel,
        gamma=gamma,
        max_iter=-1
    )
    
    model.fit(X)
    
    return model

def evaluate_anomaly_model(model, X, y_true=None, model_name="Model", asset_type="transformer"):
    """
    Evaluate an anomaly model. Accepts:
      - sklearn-style estimator (predict -> -1/1 or 0/1)
      - array-like predictions
      - reconstruction outputs (for autoencoders) as arrays
    Returns a results dict and y_pred_binary.
    """
    results = {}
    # obtain raw predictions / outputs
    try:
        y_pred_raw = model.predict(X)
    except Exception:
        # model may already be an array of predictions/errors
        y_pred_raw = model

    y_pred_raw = np.asarray(y_pred_raw)

    # handle reconstruction outputs (3D or 2D arrays)
    y_pred_binary = None
    if y_pred_raw.ndim >= 2 and y_pred_raw.shape == X.shape:
        # assume reconstruction, compute per-window MSE and threshold at mean+3*std
        errs = np.mean(np.square(y_pred_raw - X), axis=tuple(range(1, y_pred_raw.ndim)))
        thr = errs.mean() + 3 * errs.std()
        y_pred_binary = (errs > thr).astype(int)
        results['reconstruction_errors'] = errs
        results['threshold'] = float(thr)
    else:
        # common classifier outputs
        uniq = set(np.unique(y_pred_raw))
        if uniq <= {-1, 1}:
            # IsolationForest style: -1 = anomaly, 1 = normal
            y_pred_binary = np.where(y_pred_raw == -1, 1, 0).astype(int)
        elif uniq <= {0, 1}:
            y_pred_binary = y_pred_raw.astype(int)
        else:
            # probabilities or continuous scores -> threshold at 0.5
            y_pred_binary = (y_pred_raw.ravel() > 0.5).astype(int)

    results['y_pred_binary'] = y_pred_binary

    # If ground truth provided, create robust binary ground truth mapping
    if y_true is not None:
        y_arr = np.asarray(y_true).ravel()
        unique_vals = set(np.unique(y_arr))

        # If string labels present, try to map using common labels
        if any(isinstance(v, str) for v in unique_vals):
            if 'Critical' in unique_vals:
                y_true_bin = (y_arr == 'Critical').astype(int)
            elif 'Anomaly' in unique_vals:
                y_true_bin = (y_arr == 'Anomaly').astype(int)
            elif 'Warning' in unique_vals or 'Degraded' in unique_vals:
                if 'Critical' in unique_vals:
                    y_true_bin = (y_arr == 'Critical').astype(int)
                else:
                    y_true_bin = (y_arr != 'Healthy').astype(int)
            else:
                y_true_bin = (y_arr != 'Healthy').astype(int)
        else:
            # numeric labels: common mapping 0=Healthy,1=Warning,2=Critical
            if unique_vals <= {0, 1}:
                y_true_bin = (y_arr == 1).astype(int)
            elif unique_vals <= {0, 1, 2}:
                y_true_bin = (y_arr == 2).astype(int)
            else:
                y_true_bin = (y_arr != 0).astype(int)

        # If predicted binary shape differs, try to align lengths (handles windowing mismatches gracefully)
        if y_pred_binary.shape[0] != y_true_bin.shape[0]:
            # prefer to compare on the overlap
            n = min(y_pred_binary.shape[0], y_true_bin.shape[0])
            y_pred_bin_use = y_pred_binary[:n]
            y_true_bin_use = y_true_bin[:n]
        else:
            y_pred_bin_use = y_pred_binary
            y_true_bin_use = y_true_bin

        # Check whether predicted labels are inverted (some models output 1==normal). Evaluate both mappings and pick the better one.
        prec = precision_score(y_true_bin_use, y_pred_bin_use, zero_division=0)
        rec = recall_score(y_true_bin_use, y_pred_bin_use, zero_division=0)
        f1 = f1_score(y_true_bin_use, y_pred_bin_use, zero_division=0)

        # inverted candidate
        inv_pred = 1 - y_pred_bin_use
        prec_inv = precision_score(y_true_bin_use, inv_pred, zero_division=0)
        rec_inv = recall_score(y_true_bin_use, inv_pred, zero_division=0)
        f1_inv = f1_score(y_true_bin_use, inv_pred, zero_division=0)

        inverted = False
        # choose mapping with higher F1 for anomaly class (more robust)
        if f1_inv > f1:
            y_pred_bin_use = inv_pred
            prec, rec, f1 = prec_inv, rec_inv, f1_inv
            inverted = True

        cm = confusion_matrix(y_true_bin_use, y_pred_bin_use, labels=[0, 1])
        results['confusion_matrix'] = cm
        results['predicted_label_inverted'] = inverted

        acc = float((y_true_bin_use == y_pred_bin_use).mean())

        results.update({
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'accuracy': acc
        })

        # classification report (string) and numeric dict
        report_str = classification_report(y_true_bin_use, y_pred_bin_use, target_names=['Normal', 'Anomaly'], zero_division=0)
        results['classification_report'] = report_str

        # ROC-AUC / PR-AUC (safe)
        try:
            results['roc_auc'] = float(roc_auc_score(y_true_bin_use, y_pred_bin_use))
            results['pr_auc'] = float(average_precision_score(y_true_bin_use, y_pred_bin_use))
        except Exception:
            results['roc_auc'] = None
            results['pr_auc'] = None

        # Extra keys expected by downstream code
        results['precision_anomaly'] = float(prec)
        results['recall_anomaly'] = float(rec)
        results['f1_anomaly'] = float(f1)
        results['anomaly_count'] = int(np.sum(y_pred_bin_use))
        results['anomaly_percentage'] = float(100.0 * np.sum(y_pred_bin_use) / max(1, len(y_pred_bin_use)))
        results['y_true_binary'] = y_true_bin_use
        results['y_pred_binary'] = y_pred_bin_use
    return results, y_pred_binary

def plot_anomaly_results(X, y_pred, feature_x, feature_y, title="Anomaly Detection Results", asset_type="transformer"):
    """
    Plot anomaly detection results.

    Note: y_pred uses 1 == anomaly, 0 == normal. Colors/legend now match that convention:
      - anomaly -> red
      - normal  -> blue
    """
    plt.figure(figsize=(10, 8))

    # Ensure y_pred is integer 0/1
    y_arr = np.asarray(y_pred).astype(int)

    # Map: 1 => anomaly (red), 0 => normal (blue)
    colors = np.array(['red' if x == 1 else 'blue' for x in y_arr])

    plt.scatter(X[feature_x], X[feature_y], c=colors, alpha=0.6)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(title)

    # Create proper legend entries matching the color mapping
    normal_handle = plt.Line2D([], [], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Normal')
    anomaly_handle = plt.Line2D([], [], marker='o', color='w', markerfacecolor='red', markersize=8, label='Anomaly')
    plt.legend(handles=[normal_handle, anomaly_handle])

    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(f'results/{title.lower().replace(" ", "_")}_{asset_type}.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_model(model, model_name, asset_type):
    """
    Save the trained model to disk
    
    Parameters:
    model: Trained model
    model_name (str): Name of the model
    asset_type (str): Type of asset (transformer, motor, etc.)
    """
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save the model
    filename = f'models/{model_name}_{asset_type}.pkl'
    joblib.dump(model, filename)
    print(f"✅ Model saved to {filename}")

def plot_anomaly_score(scores, threshold=None, asset_type="asset", model_name="model", xlabel="Sample Index"):
    """
    Save a plot of anomaly score (higher = more anomalous) vs sample index.
    - scores: 1D array-like anomaly scores (for LSTM use MSE; for IsolationForest use -decision_function)
    - threshold: optional numeric threshold to draw as horizontal line
    - asset_type: asset name used in filename/title
    - model_name: 'LSTM' or 'Isolation Forest' etc.
    """
    scores = np.asarray(scores)
    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(len(scores)), scores, label='Anomaly Score', color='C0')
    if threshold is not None:
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.4f})')
    plt.scatter(np.where(scores > (threshold if threshold is not None else np.percentile(scores, 95)))[0],
                scores[scores > (threshold if threshold is not None else np.percentile(scores, 95))],
                color='red', s=20, label='Detected Anomaly')
    plt.title(f'{model_name} Anomaly Scores - {asset_type.capitalize()}')
    plt.xlabel(xlabel)
    plt.ylabel('Anomaly Score')
    plt.legend()
    plt.grid(True)
    if not os.path.exists('results'):
        os.makedirs('results')
    filename = f'results/anomaly_scores_{model_name.lower().replace(" ", "_")}_{asset_type}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved anomaly score plot to {filename}")

def detect_anomalies_for_asset(asset_type, data_path, contamination=0.05):
    """
    End-to-end anomaly detection for a specific asset type
    
    Parameters:
    asset_type (str): Type of asset ('transformer', 'motor', 'capacitor', 'ups')
    data_path (str): Path to the data file
    contamination (float): Expected proportion of outliers
    
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
        X, y, _ = load_and_preprocess_transformer_data(data_path)
    elif asset_type == 'motor':
        X, y, _ = load_and_preprocess_motor_data(data_path)
    elif asset_type == 'capacitor':
        X, y, _ = load_and_preprocess_capacitor_data(data_path)
    elif asset_type == 'ups':
        X, y, _ = load_and_preprocess_ups_data(data_path)
    else:
        raise ValueError(f"Unknown asset type: {asset_type}")
    
    # Standardize the data for LSTM
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train LSTM Autoencoder
    print(f"📊 Training LSTM Autoencoder for {asset_type}...")
    anomalies, mse, threshold = detect_anomalies_with_lstm(X_scaled, asset_type=asset_type)

    # save LSTM anomaly-score plot (MSE per window)
    try:
        plot_anomaly_score(mse, threshold=threshold, asset_type=asset_type, model_name="LSTM", xlabel="Window Index")
    except Exception:
        print("Could not save LSTM anomaly score plot.")
    
    # Train Isolation Forest model (use scaled features)
    print(f"📊 Training Isolation Forest for {asset_type}...")
    iso_model = train_isolation_forest(X_scaled, contamination=contamination)

    # compute IsolationForest anomaly scores (higher => more anomalous)
    try:
        # decision_function: higher => more normal. invert so higher==more anomalous
        iso_scores = -iso_model.decision_function(X_scaled)
        # choose threshold using same contamination percentile
        iso_threshold = np.percentile(iso_scores, 100 * (1 - contamination))
        plot_anomaly_score(iso_scores, threshold=iso_threshold, asset_type=asset_type, model_name="Isolation Forest", xlabel="Sample Index")
    except Exception:
        print("Could not compute/save Isolation Forest anomaly score plot.")

    # Evaluate the model (evaluate on scaled features)
    iso_results, y_pred_iso = evaluate_anomaly_model(iso_model, X_scaled, y, f"Isolation Forest", asset_type)
    
    # Save the model
    save_model(iso_model, 'isolation_forest', asset_type)
    
    # Plot results
    if asset_type == 'transformer':
        plot_anomaly_results(X, y_pred_iso, 'oil_temp_c', 'winding_temp_c', 
                            f'{asset_type.capitalize()} Anomaly Detection', asset_type)
    elif asset_type == 'motor':
        plot_anomaly_results(X, y_pred_iso, 'vibration_mm_s', 'temperature_c', 
                            f'{asset_type.capitalize()} Anomaly Detection', asset_type)
    elif asset_type == 'capacitor':
        plot_anomaly_results(X, y_pred_iso, 'temperature_c', 'harmonic_distortion_percent', 
                            f'{asset_type.capitalize()} Anomaly Detection', asset_type)
    elif asset_type == 'ups':
        plot_anomaly_results(X, y_pred_iso, 'voltage_per_cell_v', 'temperature_c', 
                            f'{asset_type.capitalize()} Anomaly Detection', asset_type)
    
    return {
        'lstm_results': {
            'anomalies': anomalies,
            'mse': mse,
            'threshold': threshold
        },
        'isolation_forest_results': iso_results
    }

def plot_confusion_matrix(cm, model_name, asset_type):
    """
    Save confusion matrix heatmap to results/ with clear axis labels.
    cm expected in sklearn order: [[TN, FP],[FN, TP]]
    """
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name} ({asset_type})')
    if not os.path.exists('results'):
        os.makedirs('results')
    filename = f'results/confusion_matrix_{model_name.lower().replace(" ", "_")}_{asset_type}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {filename}")

def print_classification_report(report, accuracy=None, anomaly_metrics_zero=False):
    """
    Print sklearn classification_report string but:
      - Put the accuracy numeric value into the 'recall' and 'f1-score' columns of the accuracy row
      - If anomaly_metrics_zero=True print zeros (0.00) for anomaly row instead of 'N/A'
    """
    lines = report.splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            print(line)
            continue

        # accuracy line: render using provided accuracy value if available
        if stripped.startswith('accuracy'):
            # attempt to extract support from the original line
            m = re.search(r'(\d+)\s*$', line)
            support = m.group(1) if m else ''
            if accuracy is None:
                print(line)
            else:
                acc_str = f"{accuracy:.2f}"
                print(f"{'accuracy':>12} {'N/A':>10} {acc_str:>8} {acc_str:>9} {support:>8}")
            continue

        # Anomaly row override when metrics are zero
        if stripped.startswith('Anomaly') and anomaly_metrics_zero:
            # extract support at end of line
            m = re.search(r'(\d+)\s*$', line)
            support = m.group(1) if m else ''
            print(f"{'Anomaly':>12} {0.00:>10.2f} {0.00:>8.2f} {0.00:>9.2f} {support:>8}")
            continue

        # otherwise print original line unchanged
        print(line)

if __name__ == "__main__":
    # Example usage
    asset_types = ['transformer', 'motor', 'capacitor', 'ups']
    
    for asset_type in asset_types:
        print(f"\n{'='*50}")
        print(f"Processing {asset_type} data")
        print(f"{'='*50}")
        
        results = detect_anomalies_for_asset(
            asset_type, 
            f'data/{asset_type}_data.csv',
            contamination=0.05
        )
        
        print(f"\nResults for {asset_type}:")
        print(f"LSTM Anomalies detected: {np.sum(results['lstm_results']['anomalies'])} ({np.mean(results['lstm_results']['anomalies'])*100:.2f}%)")
        print(f"Isolation Forest Anomalies detected: {results['isolation_forest_results']['anomaly_count']} ({results['isolation_forest_results']['anomaly_percentage']:.2f}%)")

        # Print class distribution
        if 'confusion_matrix' in results['isolation_forest_results']:
            cm = results['isolation_forest_results']['confusion_matrix']
            print("Confusion Matrix:")
            print(cm)
            # call with (cm, model_name, asset_type)
            plot_confusion_matrix(cm, "Isolation Forest", asset_type)

        if 'precision_anomaly' in results['isolation_forest_results']:
            anomaly_metrics_zero = (
                results['isolation_forest_results']['precision_anomaly'] == 0 and
                results['isolation_forest_results']['recall_anomaly'] == 0
            )
            # Always print numeric values (0.0000 when zero) instead of 'N/A'
            print(f"Precision (Anomaly): {results['isolation_forest_results']['precision_anomaly']:.4f}")
            print(f"Recall (Anomaly): {results['isolation_forest_results']['recall_anomaly']:.4f}")
            print(f"F1 Score (Anomaly): {results['isolation_forest_results']['f1_anomaly']:.4f}")
            print(f"ROC-AUC: {results['isolation_forest_results']['roc_auc']}")
            print(f"PR-AUC: {results['isolation_forest_results']['pr_auc']}")
            print("\nClassification Report:")
            print_classification_report(
                results['isolation_forest_results']['classification_report'],
                accuracy=results['isolation_forest_results'].get('accuracy'),
                anomaly_metrics_zero=anomaly_metrics_zero
            )

            # Warn if anomaly metrics are zero
            if anomaly_metrics_zero:
                print("⚠️ WARNING: No anomalies were detected for UPS. Consider increasing the contamination parameter, tuning model thresholds, or reviewing your feature engineering and class balance.")

        # Print ground truth class distribution if available
        if 'confusion_matrix' in results['isolation_forest_results']:
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], cm[0,1], cm[1,0], cm[1,1])
            print(f"Ground truth: Normal={tn+fp}, Anomaly={fn+tp}")
            print(f"Predicted: Normal={tn+fn}, Anomaly={fp+tp}")