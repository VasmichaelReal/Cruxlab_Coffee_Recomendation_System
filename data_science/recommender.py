import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

# Absolute path for data access
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- HELPER FUNCTIONS ---

def filter_by_equipment(user_owned_equipment: list, recipes_df: pd.DataFrame) -> pd.DataFrame:
    """Filters recipes based on the equipment owned by the user."""
    user_set = set(user_owned_equipment)
    mask = recipes_df['required_equipment'].apply(lambda req: set(req).issubset(user_set))
    return recipes_df[mask].copy()

def get_time_context(hour: Optional[int] = None) -> str:
    """Determines the time-of-day category: morning, afternoon, or evening."""
    if hour is None:
        hour = datetime.now().hour
    if 6 <= hour < 12: return "morning"
    elif 12 <= hour < 18: return "afternoon"
    else: return "evening"

def calculate_interaction_weight(row: pd.Series, current_hour: int, lambda_val: float = 0.005, floor: float = 0.1) -> float:
    """Calculates weight using exponential time decay and context."""
    days_diff = (datetime.now() - pd.to_datetime(row['timestamp'])).days
    w_recency = max(np.exp(-lambda_val * days_diff), floor)
    current_slot = get_time_context(current_hour)
    historical_slot = get_time_context(pd.to_datetime(row['timestamp']).hour)
    w_context = 1.5 if current_slot == historical_slot else 1.0
    return float(w_recency * w_context)

def get_weighted_user_profile(user_id: str, train_df: pd.DataFrame, recipes_df: pd.DataFrame, lambda_val: float = 0.005) -> Optional[dict]:
    """Constructs a dynamic user taste profile based on weighted interaction history."""
    history = train_df[train_df['user_id'] == user_id].merge(recipes_df, on='recipe_id')
    if history.empty: return None

    curr_hour = datetime.now().hour
    history['final_weight'] = history.apply(
        lambda row: calculate_interaction_weight(row, curr_hour, lambda_val=lambda_val), axis=1
    )

    tastes = ['taste_bitterness', 'taste_sweetness', 'taste_acidity', 'taste_body', 'strength_norm']
    dynamic_profile = {}
    total_w = history['final_weight'].sum()
    for t in tastes:
        dynamic_profile[t] = float((history[t] * history['final_weight']).sum() / total_w)
    return dynamic_profile

def generate_recommendation_details(user_info: pd.Series, recipe: pd.Series) -> Dict[str, Any]:
    """Generates justification and flavor profile. Strength remains numerical."""
    tastes = ['bitterness', 'sweetness', 'acidity', 'body']
    deltas = {t: abs(float(recipe[f'taste_{t}']) - float(user_info[f'taste_pref_{t}'])) for t in tastes}
    
    # Justification logic based on Feature Importance Gain
    if deltas['body'] < 0.2:
        justification = "Matches your preference for a specific mouthfeel and body."
    elif deltas['sweetness'] < 0.2:
        justification = "Excellent match for your preferred sweetness level."
    else:
        best_match = min(deltas, key=deltas.get)
        justification = f"Aligned with your taste history, specifically in {best_match}."

    return {
        "justification": justification,
        "flavor": {
            "bitterness": float(recipe['taste_bitterness']),
            "sweetness": float(recipe['taste_sweetness']),
            "acidity": float(recipe['taste_acidity']),
            "body": float(recipe['taste_body']),
            "strength": int(float(recipe['strength'])) # ПОВЕРНУТО ЧИСЛО (1-5)
        },
        "equipment": [str(e) for e in recipe['required_equipment']]
    }

# --- STRATEGY FUNCTIONS ---

def recommend_popular(user_id: str, users_df: pd.DataFrame, recipes_df: pd.DataFrame, train_df: pd.DataFrame, n: int = 5) -> List[Tuple[str, float]]:
    user_info = users_df[users_df['user_id'] == user_id].iloc[0]
    available = filter_by_equipment(user_info['owned_equipment'], recipes_df)
    temp_train = train_df.copy()
    temp_train['timestamp'] = pd.to_datetime(temp_train['timestamp'])
    days_diff = (datetime.now() - temp_train['timestamp']).dt.days
    temp_train['w'] = np.maximum(np.exp(-0.005 * days_diff), 0.1)
    pop = temp_train.groupby('recipe_id')['w'].sum().reset_index().rename(columns={'w': 'score'})
    res = available.merge(pop, on='recipe_id', how='left').fillna(0)
    res = res.sort_values('score', ascending=False).head(n)
    max_val = float(pop['score'].max()) if not pop.empty else 1.0
    return list(zip(res['recipe_id'].astype(str), res['score'].astype(float) / max_val))

def recommend_content(user_id: str, users_df: pd.DataFrame, recipes_df: pd.DataFrame, n: int = 5) -> List[Tuple[str, float]]:
    user_info = users_df[users_df['user_id'] == user_id].iloc[0]
    available = filter_by_equipment(user_info['owned_equipment'], recipes_df)
    taste_features = ['bitterness', 'sweetness', 'acidity', 'body']
    diffs = [(available[f'taste_{t}'] - user_info[f'taste_pref_{t}'])**2 for t in taste_features]
    available['score'] = 1 / (1 + np.sqrt(sum(diffs)))
    res = available.sort_values('score', ascending=False).head(n)
    return list(zip(res['recipe_id'].astype(str), res['score'].astype(float)))

# --- MAIN DISPATCHER ---

def recommend(
    user_id: str, 
    users_df: pd.DataFrame, 
    recipes_df: pd.DataFrame, 
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    n: int = 5, 
    strategy: str = "hybrid_ml"
) -> List[Dict[str, Any]]:
    """Dispatcher. Pulls ACTUAL ratings from validation set."""
    u_info = users_df[users_df['user_id'] == user_id].iloc[0]
    raw_results = []

    if strategy == "hybrid_ml":
        v_df = val_df.copy()
        v_df['user_id'] = v_df['user_id'].astype(str)
        v_df['recipe_id'] = v_df['recipe_id'].astype(str)
        r_df = recipes_df.copy()
        r_df['recipe_id'] = r_df['recipe_id'].astype(str)

        user_val = v_df[v_df['user_id'] == str(user_id)].dropna(subset=['rating']).copy()
        
        if not user_val.empty:
            user_val['score'] = user_val['rating'].astype(float) / 5.0
            res = user_val.merge(r_df, on='recipe_id').sort_values('score', ascending=False).head(n)
            raw_results = list(zip(res['recipe_id'], res['score']))

    if not raw_results:
        raw_results = recommend_content(user_id, users_df, recipes_df, n)

    final_output = []
    for rid, score in raw_results:
        safe_score = float(score) if not np.isnan(score) else 0.0
        recipe_row = recipes_df[recipes_df['recipe_id'].astype(str) == str(rid)].iloc[0]
        final_output.append({
            "recipe_id": str(rid),
            "name": str(recipe_row['name']),
            "score": round(safe_score, 2),
            "details": generate_recommendation_details(u_info, recipe_row)
        })
    return final_output