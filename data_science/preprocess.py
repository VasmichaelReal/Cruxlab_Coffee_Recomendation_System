import ast
import os
import numpy as np
import pandas as pd
from typing import Tuple

# Absolute path configuration for modular data access
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def load_all_data(data_path: str = DATA_DIR) -> Tuple[pd.DataFrame, ...]:
    """
    Loads all required datasets from the specified data directory.
    Expects recipes, users, and interaction history files.
    """
    recipes = pd.read_csv(os.path.join(data_path, 'recipes.csv'))
    users = pd.read_csv(os.path.join(data_path, 'users.csv'))
    train = pd.read_csv(os.path.join(data_path, 'interactions_train.csv'))
    val_warm = pd.read_csv(os.path.join(data_path, 'interactions_val.csv'))
    val_cold = pd.read_csv(os.path.join(data_path, 'interactions_val_cold.csv'))
    
    return recipes, users, train, val_warm, val_cold

def preprocess_coffee_data(
    recipes_df: pd.DataFrame, 
    users_df: pd.DataFrame, 
    train_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Handles feature normalization, scaling, and parsing of complex string structures.
    """
    # Parse list-like strings into Python lists for filtering and analysis
    list_cols_recipes = ['required_equipment', 'required_products', 'tags']
    for col in list_cols_recipes:
        if col in recipes_df.columns:
            recipes_df[col] = recipes_df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
    
    list_cols_users = ['owned_equipment', 'available_products', 'dietary_restrictions']
    for col in list_cols_users:
        if col in users_df.columns:
            users_df[col] = users_df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

    # Normalize strength features to a 0.0 - 1.0 range
    if 'strength' in recipes_df.columns:
        recipes_df['strength_norm'] = (recipes_df['strength'] - 1) / 4
    
    if 'preferred_strength' in users_df.columns:
        users_df['pref_strength_norm'] = (users_df['preferred_strength'] - 1) / 4
    
    # Global data cleaning: Remove records with missing interaction ratings
    train_df = train_df.dropna(subset=['rating'])
    
    return recipes_df, users_df, train_df

def build_features(
    interactions_df: pd.DataFrame, 
    recipes_df: pd.DataFrame, 
    users_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges datasets and engineers features required for ranking models (LightGBM).
    Calculates taste deltas and interaction matches.
    """
    # Join interaction history with recipe and user attributes
    data = interactions_df.merge(recipes_df, on='recipe_id', how='left')
    data = data.merge(users_df, on='user_id', how='left')
    
    # Calculate absolute differences (deltas) between user preferences and recipe attributes
    for taste in ['bitterness', 'sweetness', 'acidity', 'body']:
        r_col, u_col = f'taste_{taste}', f'taste_pref_{taste}'
        if r_col in data.columns and u_col in data.columns:
            data[f'delta_{taste}'] = abs(data[r_col] - data[u_col])

    # Categorical match for preferred strength
    if 'strength' in data.columns and 'preferred_strength' in data.columns:
        data['strength_match'] = (data['strength'] == data['preferred_strength']).astype(int)
    
    # Final cleanup to prevent Input contains NaN errors in scikit-learn metrics
    critical_features = [
        'taste_bitterness', 'taste_sweetness', 'taste_acidity', 'taste_body', 'strength_norm',
        'taste_pref_bitterness', 'taste_pref_sweetness', 'taste_pref_acidity', 'taste_pref_body',
        'delta_bitterness', 'delta_sweetness', 'delta_acidity', 'delta_body', 'strength_match'
    ]
    data = data.dropna(subset=[col for col in critical_features if col in data.columns])
    
    return data

if __name__ == "__main__":
    print(f"Data Source Path: {DATA_DIR}")
    r_raw, u_raw, t_raw, _, _ = load_all_data()
    recipes, users, train = preprocess_coffee_data(r_raw, u_raw, t_raw)
    
    feature_matrix = build_features(train, recipes, users)
    print(f"Process Complete. Feature Matrix Shape: {feature_matrix.shape}")