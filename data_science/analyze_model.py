import os
import joblib
import matplotlib.pyplot as plt
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

# Local imports
from preprocess import load_all_data, preprocess_coffee_data, build_features

# --- Constants ---
FEATURES = [
    'taste_bitterness', 'taste_sweetness', 'taste_acidity', 'taste_body',
    'strength_norm', 'taste_pref_bitterness', 'taste_pref_sweetness',
    'taste_pref_acidity', 'taste_pref_body', 'pref_strength_norm',
    'delta_bitterness', 'delta_sweetness', 'delta_acidity',
    'delta_body', 'strength_match'
]

RATING_THRESHOLD = 3.2
MODEL_FILENAME = 'coffee_model_optimized.pkl'
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
    """
    Loads raw data, preprocesses it, and engineers features.
    Returns a cleaned DataFrame ready for analysis.
    """
    print("Step 1: Loading and preprocessing data...")
    try:
        r_raw, u_raw, t_raw, _, _ = load_all_data()
        recipes, users, interactions = preprocess_coffee_data(r_raw, u_raw, t_raw)
        print(f"Data loaded successfully. Interactions count: {len(interactions)}")

        print("Step 2: Merging tables and engineering features...")
        # Use build_features for consistency with the training pipeline
        data = build_features(interactions, recipes, users)
        
        # Remove rows with missing values in critical columns
        data = data.dropna(subset=FEATURES + ['rating'])
        return data

    except Exception as e:
        print(f"Critical Error during data loading: {e}")
        raise


def load_model(base_dir: str):
    """
    Loads the serialized LightGBM model from the models directory.
    """
    print("Step 3: Accessing OPTIMIZED model...")
    model_path = os.path.join(base_dir, 'models', MODEL_FILENAME)
    
    print(f"Looking for model at: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"FAILED: Model file not found at {model_path}. "
            "Please run optimize_model.py first."
        )
    
    try:
        model = joblib.load(model_path)
        print(f"Model loaded: {type(model).__name__}")
        return model
    except Exception as e:
        print(f"Error loading model file: {e}")
        raise


def evaluate_metrics(model, X_test, y_test):
    """
    Calculates and prints classification metrics (ROC AUC, Report).
    """
    print("Step 4: Running inference and calculating metrics...")
    
    # Get class predictions (0 or 1)
    y_pred = model.predict(X_test)
    # Get probabilities for ROC AUC
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 40)
    print("CLASSIFICATION PERFORMANCE (Optimized Model)")
    print("=" * 40)
    
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC Score: {auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(
        y_test, 
        y_pred, 
        target_names=['Dislike (<3.2)', 'Like (>=3.2)']
    ))
    return y_pred


def generate_plots(model, X_test, y_test, y_pred, output_dir: str):
    """
    Generates and saves Feature Importance and Confusion Matrix plots.
    """
    print("\nStep 5: Generating visual reports...")
    
    # Plot 1: Feature Importance
    plt.figure(figsize=(10, 6))
    lgb.plot_importance(
        model, 
        max_num_features=15, 
        importance_type='gain', 
        precision=1, 
        height=0.5
    )
    plt.title("Feature Importance (Gain) - Optimized Model")
    plt.tight_layout()
    
    importance_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(importance_path)
    print(f" -> Feature importance saved to: {importance_path}")
    
    # Plot 2: Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=['Dislike', 'Like']
    )
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f" -> Confusion matrix saved to: {cm_path}")


def main():
    """
    Main execution routine for model diagnostics.
    """
    print("--- Analysis Started ---")
    script_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        # 1. Prepare Data
        data = load_dataset()
        
        X = data[FEATURES]
        # IMPORTANT: Since it's a classifier, target must be binary.
        # Ensure the threshold matches the training phase.
        y_true = (data['rating'] >= RATING_THRESHOLD).astype(int)
        
        print(f"Feature matrix ready. Samples: {len(X)}")
        print(f"Class balance - Positive: {y_true.sum()}, Negative: {len(y_true) - y_true.sum()}")

        # 2. Load Model
        model = load_model(script_dir)

        # 3. Split Data
        # Using a split to test on likely unseen data
        _, X_test, _, y_test = train_test_split(
            X, y_true, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE, 
            stratify=y_true
        )

        # 4. Evaluation
        y_pred = evaluate_metrics(model, X_test, y_test)

        # 5. Visualization
        generate_plots(model, X_test, y_test, y_pred, script_dir)

        print("--- Analysis Complete ---")

    except Exception as e:
        print(f"\nAnalysis failed due to error: {e}")


if __name__ == "__main__":
    main()