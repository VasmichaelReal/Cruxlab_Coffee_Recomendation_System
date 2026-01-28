import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity


def filter_by_equipment(user_owned_equipment: list, recipes_df: pd.DataFrame) -> pd.DataFrame:
    """Filters recipes where required_equipment is a subset of owned_equipment."""
    user_set = set(user_owned_equipment)
    mask = recipes_df['required_equipment'].apply(lambda req: set(req).issubset(user_set))
    return recipes_df[mask].copy()


def recommend_popular(user_id: str, users_df: pd.DataFrame, recipes_df: pd.DataFrame, 
                      train_df: pd.DataFrame, n: int = 5) -> List[Tuple[str, float]]:
    """Baseline: Recommends most popular recipes filtered by equipment."""
    user_info = users_df[users_df['user_id'] == user_id].iloc[0]
    available = filter_by_equipment(user_info['owned_equipment'], recipes_df)
    
    popularity = train_df['recipe_id'].value_counts().reset_index()
    popularity.columns = ['recipe_id', 'count']
    
    res = available.merge(popularity, on='recipe_id', how='left').fillna(0)
    res = res.sort_values('count', ascending=False).head(n)
    # Normalize popularity score for consistency
    max_count = popularity['count'].max() if not popularity.empty else 1
    return list(zip(res['recipe_id'], res['count'].astype(float) / max_count))


def recommend_content(user_id: str, users_df: pd.DataFrame, 
                      recipes_df: pd.DataFrame, n: int = 5) -> List[Tuple[str, float]]:
    """Content-based: Uses Euclidean distance. Score normalized to [0, 1]."""
    user_info = users_df[users_df['user_id'] == user_id].iloc[0]
    available = filter_by_equipment(user_info['owned_equipment'], recipes_df)
    
    taste_features = ['bitterness', 'sweetness', 'acidity', 'body']
    diffs = [(available[f'taste_{t}'] - user_info[f'taste_pref_{t}'])**2 for t in taste_features]
    
    available['distance'] = np.sqrt(sum(diffs))
    available['score'] = 1 / (1 + available['distance'])
    
    res = available.sort_values('score', ascending=False).head(n)
    return list(zip(res['recipe_id'], res['score']))


def recommend_user_hybrid(user_id: str, users_df: pd.DataFrame, recipes_df: pd.DataFrame, 
                          train_df: pd.DataFrame, n: int = 5) -> List[Tuple[str, float]]:
    """User-User Hybrid: Finds similar users via Cosine Similarity."""
    u_info = users_df[users_df['user_id'] == user_id].iloc[0]
    taste_cols = ['taste_pref_bitterness', 'taste_pref_sweetness', 'taste_pref_acidity', 'taste_pref_body']
    u_vec = u_info[taste_cols].values.reshape(1, -1)
    
    active_ids = train_df['user_id'].unique()
    others = users_df[users_df['user_id'].isin(active_ids) & (users_df['user_id'] != user_id)].copy()
    
    if others.empty: return recommend_content(user_id, users_df, recipes_df, n)

    sims = cosine_similarity(u_vec, others[taste_cols].values)[0]
    others['sim'] = sims
    top_neighbors = others.sort_values('sim', ascending=False).head(10)
    
    neighbor_recs = train_df[train_df['user_id'].isin(top_neighbors['user_id']) & (train_df['rating'] >= 4)]
    if neighbor_recs.empty: return recommend_content(user_id, users_df, recipes_df, n)
        
    rec_counts = neighbor_recs['recipe_id'].value_counts().reset_index()
    rec_counts.columns = ['recipe_id', 'score']
    # Normalize score to 0-1
    rec_counts['score'] = rec_counts['score'] / rec_counts['score'].max()
    
    available = filter_by_equipment(u_info['owned_equipment'], recipes_df)
    res = available.merge(rec_counts, on='recipe_id').sort_values('score', ascending=False).head(n)
    return list(zip(res['recipe_id'], res['score'].astype(float)))


def recommend_weighted_content(user_id: str, users_df: pd.DataFrame, 
                               recipes_df: pd.DataFrame, n: int = 5) -> list:
    """
    Research Strategy: Weighted Euclidean Distance.
    Weights represent feature importance in coffee perception.
    """
    u = users_df[users_df['user_id'] == user_id].iloc[0]
    avail = filter_by_equipment(u['owned_equipment'], recipes_df)
    
    # Define research weights
    weights = {
        'bitterness': 1.0,
        'sweetness': 1.0,
        'acidity': 1.5,   # More important
        'body': 0.7,      # Less important
        'strength': 2.0   # Most critical
    }
    
    # Calculate weighted squared differences
    diff_sum = 0
    for taste, w in weights.items():
        if taste == 'strength':
            diff = (avail['strength_norm'] - u['pref_strength_norm'])**2
        else:
            diff = (avail[f'taste_{taste}'] - u[f'taste_pref_{taste}'])**2
        diff_sum += w * diff
        
    avail['score'] = 1 / (1 + np.sqrt(diff_sum))
    res = avail.sort_values('score', ascending=False).head(n)
    return list(zip(res['recipe_id'], res['score'].astype(float)))


def recommend(user_id: str, users_df: pd.DataFrame, recipes_df: pd.DataFrame, 
              train_df: pd.DataFrame, n: int = 5, strategy: str = "hybrid_ml") -> List[Tuple[str, float]]:
    """Main entry point. Dispatches based on user state and chosen strategy."""
    user_info = users_df[users_df['user_id'] == user_id].iloc[0]
    user_history = train_df[train_df['user_id'] == user_id]
    model_path = 'models/coffee_model.pkl'

    if strategy == "user_hybrid":
        return recommend_user_hybrid(user_id, users_df, recipes_df, train_df, n)

    if not user_history.empty and strategy == "hybrid_ml" and os.path.exists(model_path):
        model = joblib.load(model_path)
        available = filter_by_equipment(user_info['owned_equipment'], recipes_df)
        X_predict = available.copy()
        
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
        
        # Predicted ratings 1-5 normalized to 0-1
        preds = model.predict(X_predict[feature_cols])
        available['score'] = np.clip(preds / 5.0, 0, 1)
        res = available.sort_values('score', ascending=False).head(n)
        return list(zip(res['recipe_id'], res['score'].astype(float)))

    return recommend_content(user_id, users_df, recipes_df, n)