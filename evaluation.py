import numpy as np
import pandas as pd
from typing import List, Dict

def calculate_ndcg(true_ratings: Dict[str, float], predicted_ids: List[str], n: int = 5) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain (NDCG) at rank N.
    
    Filters out NaN values and compares predictions against the best possible 
    ranking from the test set.
    """
    if not true_ratings or not predicted_ids:
        return 0.0

    # 1. Clean data: Remove any NaN ratings from the ground truth
    clean_true = {k: v for k, v in true_ratings.items() if not np.isnan(v)}
    if not clean_true:
        return 0.0

    # 2. Calculate DCG@n (Discounted Cumulative Gain)
    # Relevance is normalized (rating / 5.0)
    dcg = 0.0
    for i, rid in enumerate(predicted_ids[:n]):
        relevance = float(clean_true.get(rid, 0.0)) / 5.0
        dcg += relevance / np.log2(i + 2)

    # 3. Calculate IDCG@n (Ideal DCG)
    # Based on the best possible items available in the test set
    ideal_ratings = sorted(clean_true.values(), reverse=True)[:n]
    idcg = sum([float(rel) / 5.0 / np.log2(i + 2) for i, rel in enumerate(ideal_ratings)])

    return dcg / idcg if idcg > 0 else 0.0

def get_user_relevance(user_id: str, recommended_ids: List[str], val_df: pd.DataFrame) -> List[float]:
    """
    Maps recommended IDs to normalized ground truth relevance scores (0.0 - 1.0).
    """
    user_val = val_df[val_df['user_id'] == user_id]
    relevances = []
    
    for rid in recommended_ids:
        match = user_val[user_val['recipe_id'] == rid]
        if not match.empty and not pd.isna(match.iloc[0]['rating']):
            relevances.append(float(match.iloc[0]['rating']) / 5.0)
        else:
            relevances.append(0.0)
            
    return relevances