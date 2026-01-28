import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity

# Absolute path for model and data access
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def filter_by_equipment(user_owned_equipment: list, recipes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters recipes based on the equipment owned by the user.
    """
    user_set = set(user_owned_equipment)
    mask = recipes_df['required_equipment'].apply(lambda req: set(req).issubset(user_set))
    return recipes_df[mask].copy()

def get_time_context(hour: Optional[int] = None) -> str:
    """
    Determines the time-of-day category: morning, afternoon, or evening.
    """
    if hour is None:
        hour = datetime.now().hour
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    else:
        return "evening"

def calculate_interaction_weight(
    row: pd.Series, 
    current_hour: int, 
    lambda_val: float = 0.005, 
    floor: float = 0.1
) -> float:
    """
    Calculates the combined weight of an interaction using exponential 
    time decay and time-of-day context.
    """
    # Recency decay calculation
    days_diff = (datetime.now() - pd.to_datetime(row['timestamp'])).days
    w_recency = max(np.exp(-lambda_val * days_diff), floor)
    
    # Time-of-day context multiplier
    current_slot = get_time_context(current_hour)
    historical_slot = get_time_context(pd.to_datetime(row['timestamp']).hour)
    w_context = 1.5 if current_slot == historical_slot else 1.0
    
    return w_recency * w_context

def get_weighted_user_profile(
    user_id: str, 
    train_df: pd.DataFrame, 
    recipes_df: pd.DataFrame, 
    lambda_val: float = 0.005
) -> Optional[dict]:
    """
    Constructs a dynamic user taste profile based on weighted interaction history.
    """
    history = train_df[train_df['user_id'] == user_id].merge(recipes_df, on='recipe_id')
    if history.empty:
        return None

    curr_hour = datetime.now().hour
    history['final_weight'] = history.apply(
        lambda row: calculate_interaction_weight(row, curr_hour, lambda_val=lambda_val), 
        axis=1
    )

    tastes = ['taste_bitterness', 'taste_sweetness', 'taste_acidity', 'taste_body', 'strength_norm']
    dynamic_profile = {}
    total_w = history['final_weight'].sum()
    
    for t in tastes:
        dynamic_profile[t] = (history[t] * history['final_weight']).sum() / total_w
        
    return dynamic_profile

def recommend_popular(
    user_id: str, 
    users_df: pd.DataFrame, 
    recipes_df: pd.DataFrame, 
    train_df: pd.DataFrame, 
    n: int = 5, 
    precalculated_pop: Optional[pd.DataFrame] = None
) -> List[Tuple[str, float]]:
    """
    Recommends globally popular recipes weighted by recency and context.
    """
    user_info = users_df[users_df['user_id'] == user_id].iloc[0]
    available = filter_by_equipment(user_info['owned_equipment'], recipes_df)
    
    if precalculated_pop is not None:
        pop = precalculated_pop
    else:
        temp_train = train_df.copy()
        temp_train['timestamp'] = pd.to_datetime(temp_train['timestamp'])
        days_diff = (datetime.now() - temp_train['timestamp']).dt.days
        temp_train['w'] = np.maximum(np.exp(-0.005 * days_diff), 0.1)
        pop = temp_train.groupby('recipe_id')['w'].sum().reset_index().rename(columns={'w': 'score'})

    res = available.merge(pop, on='recipe_id', how='left').fillna(0)
    res = res.sort_values('score', ascending=False).head(n)
    
    max_val = pop['score'].max() if not pop.empty else 1
    return list(zip(res['recipe_id'], res['score'].astype(float) / max_val))

def recommend_weighted_content(
    user_id: str, 
    users_df: pd.DataFrame, 
    recipes_df: pd.DataFrame, 
    train_df: pd.DataFrame, 
    n: int = 5, 
    lambda_val: float = 0.001
) -> List[Tuple[str, float]]:
    """
    Personalized research strategy using weighted Euclidean distance and dynamic profiles.
    Parameters optimized via Random Search: Lambda=0.001.
    """
    u_static = users_df[users_df['user_id'] == user_id].iloc[0]
    avail = filter_by_equipment(u_static['owned_equipment'], recipes_df)
    
    dynamic_profile = get_weighted_user_profile(user_id, train_df, recipes_df, lambda_val=lambda_val)
    
    # Optimized feature weights
    weights = {'bitterness': 2.0, 'sweetness': 0.5, 'acidity': 2.0, 'body': 0.5, 'strength': 2.0}
    
    diff_sum = 0
    for taste, w in weights.items():
        if dynamic_profile:
            u_val = dynamic_profile['strength_norm'] if taste == 'strength' else dynamic_profile[f'taste_{taste}']
        else:
            u_val = u_static['pref_strength_norm'] if taste == 'strength' else u_static[f'taste_pref_{taste}']
            
        r_val = avail['strength_norm'] if taste == 'strength' else avail[f'taste_{taste}']
        diff_sum += w * (r_val - u_val)**2
        
    avail['score'] = 1 / (1 + np.sqrt(diff_sum))
    res = avail.sort_values('score', ascending=False).head(n)
    return list(zip(res['recipe_id'], res['score'].astype(float)))

def recommend_content(
    user_id: str, 
    users_df: pd.DataFrame, 
    recipes_df: pd.DataFrame, 
    n: int = 5
) -> List[Tuple[str, float]]:
    """
    Standard content-based recommendation using static profile and Euclidean distance.
    """
    user_info = users_df[users_df['user_id'] == user_id].iloc[0]
    available = filter_by_equipment(user_info['owned_equipment'], recipes_df)
    taste_features = ['bitterness', 'sweetness', 'acidity', 'body']
    diffs = [(available[f'taste_{t}'] - user_info[f'taste_pref_{t}'])**2 for t in taste_features]
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
    User-User collaborative filtering based on profile similarity.
    """
    u_info = users_df[users_df['user_id'] == user_id].iloc[0]
    taste_cols = ['taste_pref_bitterness', 'taste_pref_sweetness', 'taste_pref_acidity', 'taste_pref_body']
    u_vec = u_info[taste_cols].values.reshape(1, -1)
    
    active_ids = train_df['user_id'].unique()
    others = users_df[users_df['user_id'].isin(active_ids) & (users_df['user_id'] != user_id)].copy()
    
    if others.empty: 
        return recommend_content(user_id, users_df, recipes_df, n)
        
    sims = cosine_similarity(u_vec, others[taste_cols].values)[0]
    others['sim'] = sims
    top_neighbors = others.sort_values('sim', ascending=False).head(10)
    
    neighbor_recs = train_df[train_df['user_id'].isin(top_neighbors['user_id']) & (train_df['rating'] >= 4)]
    if neighbor_recs.empty: 
        return recommend_content(user_id, users_df, recipes_df, n)
        
    rec_counts = neighbor_recs['recipe_id'].value_counts().reset_index()
    rec_counts.columns = ['recipe_id', 'score']
    rec_counts['score'] = rec_counts['score'] / rec_counts['score'].max()
    
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
    Primary dispatcher for recommendation strategies.
    """
    if strategy == "popularity":
        return recommend_popular(user_id, users_df, recipes_df, train_df, n)
    if strategy == "user_hybrid":
        return recommend_user_hybrid(user_id, users_df, recipes_df, train_df, n)
    if strategy == "weighted_content":
        return recommend_weighted_content(user_id, users_df, recipes_df, train_df, n)

    model_path = os.path.join(BASE_DIR, 'models', 'coffee_model.pkl')
    user_history = train_df[train_df['user_id'] == user_id]
    
    if strategy == "hybrid_ml" and not user_history.empty and os.path.exists(model_path):
        model = joblib.load(model_path)
        u_info = users_df[users_df['user_id'] == user_id].iloc[0]
        avail = filter_by_equipment(u_info['owned_equipment'], recipes_df)
        X = avail.copy()
        
        for t in ['bitterness', 'sweetness', 'acidity', 'body']:
            X[f'taste_pref_{t}'] = u_info[f'taste_pref_{t}']
            X[f'delta_{t}'] = abs(X[f'taste_{t}'] - u_info[f'taste_pref_{t}'])
            
        X['pref_strength_norm'] = u_info['pref_strength_norm']
        X['strength_match'] = (X['strength'] == u_info['preferred_strength']).astype(int)
        
        cols = [
            'taste_bitterness', 'taste_sweetness', 'taste_acidity', 'taste_body', 'strength_norm',
            'taste_pref_bitterness', 'taste_pref_sweetness', 'taste_pref_acidity', 'taste_pref_body', 
            'pref_strength_norm', 'delta_bitterness', 'delta_sweetness', 'delta_acidity', 'delta_body', 'strength_match'
        ]
        
        avail['score'] = np.clip(model.predict(X[cols]) / 5.0, 0, 1)
        res = avail.sort_values('score', ascending=False).head(n)
        return list(zip(res['recipe_id'], res['score'].astype(float)))

    return recommend_content(user_id, users_df, recipes_df, n)