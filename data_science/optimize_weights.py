import os
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import optuna
from tqdm import tqdm

# Local module imports
from preprocess import load_all_data, preprocess_coffee_data
from evaluation import calculate_ndcg
from recommender import filter_by_equipment

# --- Constants ---
TASTE_FEATURES = ['bitterness', 'sweetness', 'acidity', 'body', 'strength']
FEATURE_COLS = [
    'taste_bitterness', 
    'taste_sweetness', 
    'taste_acidity', 
    'taste_body', 
    'strength_norm'
]
TOP_K_USERS_FOR_VALIDATION = 100  # Number of users to use for fast validation
N_TRIALS = 50


def prepare_interaction_data(
    train_df: pd.DataFrame, 
    recipes_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepares interaction data for vectorized calculations.

    Steps:
    1. Merges user interaction history with recipe details.
    2. Calculates temporal features: 'days_diff' (recency) and 'context_w' (time of day relevance).

    Args:
        train_df: DataFrame containing user interactions (ratings, timestamps).
        recipes_df: DataFrame containing recipe attributes.

    Returns:
        pd.DataFrame: Merged DataFrame with additional temporal weight columns.
    """
    # Merge interaction history with recipe attributes
    full_history = train_df.merge(recipes_df, on='recipe_id')
    
    # Calculate days difference from now
    now = datetime.now()
    full_history['days_diff'] = (now - pd.to_datetime(full_history['timestamp'])).dt.days
    
    # Vectorized time slot determination
    hours = pd.to_datetime(full_history['timestamp']).dt.hour
    
    # Define current time slot
    current_h = now.hour
    if 6 <= current_h < 12:
        current_slot = 0  # Morning
    elif 12 <= current_h < 18:
        current_slot = 1  # Afternoon
    else:
        current_slot = 2  # Evening

    # Map interaction hours to slots (0=Morning, 1=Afternoon, 2=Evening)
    conditions = [
        (hours >= 6) & (hours < 12),
        (hours >= 12) & (hours < 18)
    ]
    choices = [0, 1]
    interaction_slots = np.select(conditions, choices, default=2)
    
    # Assign context weight: 1.5 if slots match, 1.0 otherwise
    full_history['context_w'] = np.where(interaction_slots == current_slot, 1.5, 1.0)
    
    return full_history


def calculate_user_profiles(
    history_df: pd.DataFrame, 
    lambda_val: float
) -> pd.DataFrame:
    """
    Calculates weighted user preference profiles based on interaction history.
    
    Uses exponential decay (lambda) to weigh recent interactions higher.
    
    Args:
        history_df: Pre-processed history DataFrame with 'days_diff' and 'context_w'.
        lambda_val: Decay factor for recency.

    Returns:
        pd.DataFrame: A DataFrame indexed by user_id containing weighted average taste features.
    """
    # Calculate recency weight: exp(-lambda * days)
    recency_w = np.exp(-lambda_val * history_df['days_diff'])
    
    # Combine recency and context weights (with a minimum floor of 0.01)
    final_w = np.maximum(recency_w, 0.01) * history_df['context_w']
    
    # Create a temporary DF for weighted aggregation
    # Multiply taste features by the calculated weight
    weighted_features = history_df[FEATURE_COLS].multiply(final_w, axis=0)
    weighted_features['user_id'] = history_df['user_id']
    weighted_features['weight'] = final_w
    
    # Aggregation: Sum of weighted features and sum of weights
    grouped = weighted_features.groupby('user_id')
    sum_features = grouped[FEATURE_COLS].sum()
    sum_weights = grouped['weight'].sum()
    
    # Normalize to get the weighted average profile
    profiles_df = sum_features.div(sum_weights, axis=0)
    
    return profiles_df


def objective(
    trial: optuna.Trial,
    history_base: pd.DataFrame,
    test_users: np.ndarray,
    user_equipment_map: Dict,
    true_ratings_map: Dict,
    recipes: pd.DataFrame
) -> float:
    """
    Optuna objective function to optimize heuristic weights.
    
    Search Space:
    - lambda: Temporal decay rate.
    - w_*: Importance weights for each taste feature.
    """
    # --- 1. Define Hyperparameters (Search Space) ---
    lambda_val = trial.suggest_float("lambda", 0.0001, 0.1, log=True)
    
    weights = {}
    for feature in TASTE_FEATURES:
        weights[feature] = trial.suggest_float(f"w_{feature}", 0.0, 3.0)

    # --- 2. Calculate User Profiles ---
    profiles_df = calculate_user_profiles(history_base, lambda_val)
    
    # --- 3. Evaluation Loop ---
    scores = []
    
    # Convert weights dict to array in the correct order for vectorization
    w_vector = np.array([weights[f] for f in TASTE_FEATURES])

    for user_id in test_users:
        if user_id not in profiles_df.index or user_id not in user_equipment_map:
            continue
            
        # Get user profile vector [bitter, sweet, acid, body, strength]
        user_profile_vector = profiles_df.loc[user_id].values
        
        # Filter recipes by owned equipment
        owned_eq = user_equipment_map[user_id]
        available = filter_by_equipment(owned_eq, recipes)
        
        if available.empty:
            continue

        # Extract recipe feature matrix (N_recipes x 5_features)
        recipe_matrix = available[FEATURE_COLS].values
        
        # Calculate Weighted Euclidean Distance: Sum(W * (R - U)^2)
        diff_sq = (recipe_matrix - user_profile_vector) ** 2
        weighted_dist = np.sum(diff_sq * w_vector, axis=1)
        
        # Calculate Score (Inverse distance)
        # Use copy to avoid SettingWithCopyWarning on the slice
        available_scored = available.copy()
        available_scored['score'] = 1 / (1 + np.sqrt(weighted_dist))
        
        # Get top 5 recommendations
        top_recs = available_scored.nlargest(5, 'score')['recipe_id'].tolist()
        
        # Calculate Metric (NDCG)
        true_ratings = true_ratings_map.get(user_id, {})
        if true_ratings:
            scores.append(calculate_ndcg(true_ratings, top_recs, n=5))
    
    # Return mean score (maximize this)
    return np.mean(scores) if scores else 0.0


def main():
    """
    Main execution routine for hyperparameter optimization using Optuna.
    """
    # 1. Load Data
    print("Loading datasets...")
    recipes_raw, users_raw, train_raw, test_raw, _ = load_all_data()
    
    # Filter test data to only include valid ratings
    test_raw = test_raw.dropna(subset=['rating'])
    
    print("Preprocessing data...")
    recipes, users, train = preprocess_coffee_data(recipes_raw, users_raw, train_raw)
    
    # 2. Prepare Base Interaction Data
    # Computed once to avoid overhead inside the optimization loop
    print("Preparing base interaction data (temporal features)...")
    history_base = prepare_interaction_data(train, recipes)
    
    # 3. Cache Data for Optimization
    # Use a subset of users for faster validation during optimization
    test_users = test_raw['user_id'].unique()[:TOP_K_USERS_FOR_VALIDATION]
    
    user_equipment_map = dict(zip(users['user_id'], users['owned_equipment']))
    
    # Pre-build a map of ground truth ratings for O(1) access
    true_ratings_map = {
        uid: dict(zip(
            test_raw[test_raw['user_id'] == uid]['recipe_id'], 
            test_raw[test_raw['user_id'] == uid]['rating']
        ))
        for uid in test_users
    }

    # 4. Run Optuna Optimization
    print("\nStarting Optuna Optimization...")
    
    # Use partial or lambda to inject static data into the objective function
    study = optuna.create_study(direction="maximize", study_name="Heuristic_Weights_Tuning")
    
    study.optimize(
        lambda trial: objective(
            trial, 
            history_base, 
            test_users, 
            user_equipment_map, 
            true_ratings_map, 
            recipes
        ), 
        n_trials=N_TRIALS
    )

    # 5. Output Results
    print("\n" + "=" * 40)
    print("OPTIMIZATION COMPLETE")
    print(f"Best NDCG: {study.best_value:.4f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v:.4f}")
    print("=" * 40)

if __name__ == "__main__":
    main()