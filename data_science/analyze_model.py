import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from preprocess import load_all_data, preprocess_coffee_data

def diagnostic_report():
    """
    Performs model evaluation and feature importance analysis.
    Logs every step to prevent silent failures.
    """
    print("--- Analysis Started ---")
    
    # 1. Data Loading
    print("Step 1: Loading raw datasets...")
    try:
        r_raw, u_raw, t_raw, _, _ = load_all_data()
        recipes, users, interactions = preprocess_coffee_data(r_raw, u_raw, t_raw)
        print(f"Data loaded successfully. Interactions count: {len(interactions)}")
    except Exception as e:
        print(f"Critical Error during data loading: {e}")
        return

    # 2. Feature Engineering
    print("Step 2: Merging tables and engineering features...")
    data = interactions.merge(users, on='user_id').merge(recipes, on='recipe_id')
    
    for taste in ['bitterness', 'sweetness', 'acidity', 'body']:
        data[f'delta_{taste}'] = abs(data[f'taste_{taste}'] - data[f'taste_pref_{taste}'])
    data['strength_match'] = (data['strength'] == data['preferred_strength']).astype(int)

    features = [
        'taste_bitterness', 'taste_sweetness', 'taste_acidity', 'taste_body', 'strength_norm',
        'taste_pref_bitterness', 'taste_pref_sweetness', 'taste_pref_acidity', 'taste_pref_body', 
        'pref_strength_norm', 'delta_bitterness', 'delta_sweetness', 'delta_acidity', 'delta_body', 'strength_match'
    ]
    
    data = data.dropna(subset=features + ['rating'])
    X = data[features]
    y = data['rating']
    print(f"Feature matrix ready. Samples after cleaning: {len(X)}")

    # 3. Model Loading
    print("Step 3: Accessing serialized model...")
    # Using absolute paths to ensure reliability
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'models', 'coffee_model.pkl')
    
    print(f"Looking for model at: {model_path}")
    if not os.path.exists(model_path):
        print(f"FAILED: Model file not found. Ensure you have run train_model.py first.")
        return
    
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model file: {e}")
        return

    # 4. Evaluation
    print("Step 4: Running inference and calculating metrics...")
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    predictions = model.predict(X_test)

    print("\n" + "="*40)
    print("REGRESSION PERFORMANCE")
    print(f"MAE:  {mean_absolute_error(y_test, predictions):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.4f}")

    print("\n" + "="*40)
    print("CLASSIFICATION REPORT (Threshold 3.2)")
    y_true_bin = (y_test >= 4).astype(int)
    y_pred_bin = (predictions >= 3.2).astype(int)
    print(classification_report(y_true_bin, y_pred_bin, target_names=['Dislike', 'Like']))

    # 5. Visualization
    print("\nStep 5: Generating feature importance plot...")
    plt.figure(figsize=(10, 8))
    lgb.plot_importance(model, max_num_features=15, importance_type='gain', precision=1)
    plt.title("Feature Importance Analysis (Gain)")
    
    plot_path = os.path.join(script_dir, 'feature_importance.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Visualization saved to: {plot_path}")
    print("--- Analysis Complete ---")

if __name__ == "__main__":
    diagnostic_report()