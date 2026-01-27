import ast
import os
import pandas as pd
import numpy as np
from typing import Tuple


def load_all_data(data_path: str = 'data/') -> Tuple[pd.DataFrame, ...]:
    """
    Load all CSV datasets required for training and evaluation.
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
    Clean raw data, parse JSON-like strings, and normalize scores.
    """
    # Parse string representations of lists into actual Python lists
    recipe_list_cols = ['required_equipment', 'required_products', 'tags']
    for col in recipe_list_cols:
        recipes_df[col] = recipes_df[col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    
    user_list_cols = ['owned_equipment', 'available_products', 'dietary_restrictions']
    for col in user_list_cols:
        users_df[col] = users_df[col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

    # Implicit feedback: assign 3.0 to completed recipes without a rating
    train_df['final_score'] = train_df['rating'].fillna(
        train_df['completed'].map({True: 3.0, False: 0.0})
    )
    
    # Min-Max Normalization for strength (1-5 scale to 0-1)
    recipes_df['strength_norm'] = (recipes_df['strength'] - 1) / 4
    users_df['pref_strength_norm'] = (users_df['preferred_strength'] - 1) / 4
    
    return recipes_df, users_df, train_df


def build_features(
    interactions_df: pd.DataFrame, 
    recipes_df: pd.DataFrame, 
    users_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a feature matrix for the ranking model by merging data and 
    calculating taste deltas.
    """
    # Merge datasets to combine user preferences with recipe profiles
    data = interactions_df.merge(recipes_df, on='recipe_id', how='left')
    data = data.merge(users_df, on='user_id', how='left')
    
    # Calculate absolute differences between recipe profile and user preferences
    for taste in ['bitterness', 'sweetness', 'acidity', 'body']:
        recipe_col = f'taste_{taste}'
        user_col = f'taste_pref_{taste}'
        data[f'delta_{taste}'] = abs(data[recipe_col] - data[user_col])

    # Feature indicating exact strength preference match
    data['strength_match'] = (data['strength'] == data['preferred_strength']).astype(int)
    
    return data


if __name__ == "__main__":
    # Test the preprocessing pipeline
    r_raw, u_raw, t_raw, v_warm, v_cold = load_all_data()
    recipes, users, train = preprocess_coffee_data(r_raw, u_raw, t_raw)
    
    training_matrix = build_features(train, recipes, users)
    print(f"Feature matrix generated successfully. Shape: {training_matrix.shape}")