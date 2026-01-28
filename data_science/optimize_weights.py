import random
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Any

from preprocess import load_all_data, preprocess_coffee_data
from evaluation import calculate_ndcg
from recommender import filter_by_equipment

def precalculate_all_profiles(
    lambda_space: List[float], 
    train_df: pd.DataFrame, 
    recipes_df: pd.DataFrame
) -> Dict[float, Dict[str, Dict[str, float]]]:
    """
    Vectorized pre-calculation of user profiles across the hyperparameter space.
    Minimizes redundant computation by merging datasets once and using 
    batch processing for decay weights.
    """
    # Merge interaction history with recipe attributes
    full_history = train_df.merge(recipes_df, on='recipe_id')
    
    # Temporal feature engineering
    now = datetime.now()
    full_history['days_diff'] = (now - pd.to_datetime(full_history['timestamp'])).dt.days
    full_history['hour'] = pd.to_datetime(full_history['timestamp']).dt.hour
    
    # Determine temporal context slots
    def get_time_slot(h: int) -> str:
        if 6 <= h < 12:
            return "morning"
        elif 12 <= h < 18:
            return "afternoon"
        return "evening"
    
    current_slot = get_time_slot(now.hour)
    full_history['slot'] = full_history['hour'].apply(get_time_slot)
    full_history['context_w'] = np.where(full_history['slot'] == current_slot, 1.5, 1.0)

    cache = {}
    taste_features = ['taste_bitterness', 'taste_sweetness', 'taste_acidity', 'taste_body', 'strength_norm']

    for lambda_val in lambda_space:
        # Calculate recency weights with a minimum floor of 0.1
        full_history['recency_w'] = np.maximum(np.exp(-lambda_val * full_history['days_diff']), 0.1)
        full_history['final_w'] = full_history['recency_w'] * full_history['context_w']
        
        # Batch profile generation via group aggregation
        profiles = {}
        grouped = full_history.groupby('user_id')
        
        for user_id, group in grouped:
            total_weight = group['final_w'].sum()
            user_profile = {}
            for feature in taste_features:
                user_profile[feature] = (group[feature] * group['final_w']).sum() / total_weight
            profiles[user_id] = user_profile
        
        cache[lambda_val] = profiles
    return cache

def main():
    """
    Main execution routine for hyperparameter optimization using Randomized Search.
    Optimizes feature weights and the temporal decay constant (Lambda).
    """
    # Load and preprocess datasets
    recipes_raw, users_raw, train_raw, test_raw, _ = load_all_data()
    test_raw = test_raw.dropna(subset=['rating'])
    recipes, users, train = preprocess_coffee_data(recipes_raw, users_raw, train_raw)

    # Search space definition
    lambda_space = [0.001, 0.005, 0.01, 0.02]
    weight_options = [0.5, 1.0, 2.0]
    taste_features = ['bitterness', 'sweetness', 'acidity', 'body', 'strength']
    n_trials = 100 

    # Profile pre-calculation phase
    profile_cache = precalculate_all_profiles(lambda_space, train, recipes)

    best_ndcg = -1.0
    best_params = {}

    # Optimization loop optimization
    test_users = test_raw['user_id'].unique()
    user_equipment_map = dict(zip(users['user_id'], users['owned_equipment']))
    
    for _ in tqdm(range(n_trials), desc="Hyperparameter Optimization"):
        current_lambda = random.choice(lambda_space)
        current_weights = {feature: random.choice(weight_options) for feature in taste_features}
        active_profiles = profile_cache[current_lambda]
        
        trial_scores = []
        for user_id in test_users:
            if user_id not in active_profiles or user_id not in user_equipment_map:
                continue
            
            user_test_set = test_raw[test_raw['user_id'] == user_id]
            owned_equipment = user_equipment_map[user_id]
            available_recipes = filter_by_equipment(owned_equipment, recipes)
            
            if available_recipes.empty:
                continue
            
            dynamic_profile = active_profiles[user_id]
            
            # Weighted Euclidean distance vectorization
            squared_diff_sum = 0
            for feature, weight in current_weights.items():
                target_col = 'strength_norm' if feature == 'strength' else f'taste_{feature}'
                u_val = dynamic_profile[target_col]
                r_val = available_recipes[target_col]
                squared_diff_sum += weight * (r_val - u_val)**2
            
            available_recipes['score'] = 1 / (1 + np.sqrt(squared_diff_sum))
            top_recommendations = available_recipes.sort_values(
                'score', ascending=False
            ).head(5)['recipe_id'].tolist()
            
            true_ratings = dict(zip(user_test_set['recipe_id'], user_test_set['rating']))
            trial_scores.append(calculate_ndcg(true_ratings, top_recommendations, n=5))
            
        mean_ndcg = np.mean(trial_scores) if trial_scores else 0.0
        
        if mean_ndcg > best_ndcg:
            best_ndcg = mean_ndcg
            best_params = {
                "weights": current_weights, 
                "lambda": current_lambda
            }

    # Technical output reporting
    print("\n" + "="*40)
    print(f"OPTIMIZATION SUMMARY")
    print(f"Best Mean NDCG@5: {best_ndcg:.4f}")
    print(f"Optimal Lambda: {best_params.get('lambda')}")
    print(f"Optimal Feature Weights: {best_params.get('weights')}")
    print("="*40)

if __name__ == "__main__":
    main()