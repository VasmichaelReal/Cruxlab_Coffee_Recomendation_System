import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity


def filter_by_equipment(user_owned_equipment: list, recipes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters recipes where required_equipment is a subset of owned_equipment.
    """
    user_set = set(user_owned_equipment)
    mask = recipes_df['required_equipment'].apply(lambda req: set(req).issubset(user_set))
    return recipes_df[mask].copy()


def recommend_popular(
    user_id: str, 
    users_df: pd.DataFrame, 
    recipes_df: pd.DataFrame, 
    train_df: pd.DataFrame, 
    n: int = 5
) -> List[Tuple[str, float]]:
    """
    Strategy 0: Popularity baseline. High-level fallback.
    """
    u_info = users_df[users_df['user_id'] == user_id].iloc[0]
    available = filter_by_equipment(u_info['owned_equipment'], recipes_df)
    
    pop = train_df['recipe_id'].value_counts().reset_index()
    pop.columns = ['recipe_id', 'count']
    
    res = available.merge(pop, on='recipe_id', how='left').fillna(0)
    res = res.sort_values('count', ascending=False).head(n)
    return list(zip(res['recipe_id'], res['count'].astype(float)))


def recommend_content(
    user_id: str, 
    users_df: pd.DataFrame, 
    recipes_df: pd.DataFrame, 
    n: int = 5
) -> List[Tuple[str, float]]:
    """
    Strategy 1: Direct Content-Based (Euclidean). 
    Matches user profile directly to recipe attributes.
    """
    u_info = users_df[users_df['user_id'] == user_id].iloc[0]
    available = filter_by_equipment(u_info['owned_equipment'], recipes_df)
    
    taste_features = ['bitterness', 'sweetness', 'acidity', 'body']
    diffs = []
    for t in taste_features:
        # Euclidean component: $(u_i - v_i)^2$
        diff = (available[f'taste_{t}'] - u_info[f'taste_pref_{t}'])**2
        diffs.append(diff)
    
    available['distance'] = np.sqrt(sum(diffs))
    available['score'] = 1 / (1 + available['distance'])
    
    res = available.sort_values('score', ascending=False).head(n)
    return list(zip(res['recipe_id'], res['score']))


def recommend_user_hybrid(
    user_id: str, 
    users_df: pd.DataFrame, 
    recipes_df: pd.DataFrame, 
    train_df: pd.DataFrame, 
    n: int = 5
) -> List[Tuple[str, float]]:
    """
    Strategy 2: User-User Hybrid (Cosine).
    Finds similar users by taste profile and recommends what they liked.
    """
    u_info = users_df[users_df['user_id'] == user_id].iloc[0]
    taste_cols = ['taste_pref_bitterness', 'taste_pref_sweetness', 'taste_pref_acidity', 'taste_pref_body']
    
    # Vector for the current user
    u_vec = u_info[taste_cols].values.reshape(1, -1)
    
    # Other users who have ratings
    active_ids = train_df['user_id'].unique()
    others = users_df[users_df['user_id'].isin(active_ids) & (users_df['user_id'] != user_id)].copy()
    
    if others.empty:
        return recommend_content(user_id, users_df, recipes_df, n)

    # Cosine Similarity: $S_c(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$
    other_vecs = others[taste_cols].values
    sims = cosine_similarity(u_vec, other_vecs)[0]
    others['sim'] = sims
    
    # Get top 10 similar users
    top_neighbors = others.sort_values('sim', ascending=False).head(10)
    
    # Get recipes liked by these neighbors (rating >= 4)
    neighbor_recs = train_df[train_df['user_id'].isin(top_neighbors['user_id']) & (train_df['rating'] >= 4)]
    
    if neighbor_recs.empty:
        return recommend_content(user_id, users_df, recipes_df, n)
        
    # Aggregate and filter by equipment
    rec_counts = neighbor_recs['recipe_id'].value_counts().reset_index()
    rec_counts.columns = ['recipe_id', 'score']
    
    available = filter_by_equipment(u_info['owned_equipment'], recipes_df)
    res = available.merge(rec_counts, on='recipe_id').sort_values('score', ascending=False).head(n)
    
    return list(zip(res['recipe_id'], res['score'].astype(float)))


def recommend(
    user_id: str,
    users_df: pd.DataFrame,
    recipes_df: pd.DataFrame,
    train_df: pd.DataFrame,
    n: int = 5,
    strategy: str = "hybrid_ml"
) -> List[Tuple[str, float]]:
    """
    Master Recommender: Dispatches requests to specific strategies.
    """
    user_info = users_df[users_df['user_id'] == user_id].iloc[0]
    available_recipes = filter_by_equipment(user_info['owned_equipment'], recipes_df)
    
    if available_recipes.empty:
        return []

    # 1. User-User Hybrid Strategy (Your new idea)
    if strategy == "user_hybrid":
        return recommend_user_hybrid(user_id, users_df, recipes_df, train_df, n)

    # 2. Content-Based Only Strategy
    if strategy == "content_only":
        return recommend_content(user_id, users_df, recipes_df, n)

    # 3. Hybrid ML (LightGBM) - Default for warm users
    user_history = train_df[train_df['user_id'] == user_id]
    model_path = 'models/coffee_model.pkl'

    if not user_history.empty and os.path.exists(model_path):
        model = joblib.load(model_path)
        X_predict = available_recipes.copy()
        
        # Feature Engineering for LightGBM
        for t in ['bitterness', 'sweetness', 'acidity', 'body']:
            X_predict[f'taste_pref_{t}'] = user_info[f'taste_pref_{t}']
            X_predict[f'delta_{t}'] = abs(X_predict[f'taste_{t}'] - user_info[f'taste_pref_{t}'])
        
        X_predict['pref_strength_norm'] = user_info['pref_strength_norm']
        X_predict['strength_match'] = (X_predict['strength'] == user_info['preferred_strength']).astype(int)
        
        feature_cols = [
            'taste_bitterness', 'taste_sweetness', 'taste_acidity', 'taste_body', 'strength_norm',
            'taste_pref_bitterness', 'taste_pref_sweetness', 'taste_pref_acidity', 'taste_pref_body', 
            'pref_strength_norm', 'delta_bitterness', 'delta_sweetness', 'delta_acidity', 'delta_body', 
            'strength_match'
        ]
        
        available_recipes['score'] = model.predict(X_predict[feature_cols])
        res = available_recipes.sort_values('score', ascending=False).head(n)
        return list(zip(res['recipe_id'], res['score'].astype(float)))

    # Fallback for Cold Start in ML strategy
    return recommend_content(user_id, users_df, recipes_df, n)