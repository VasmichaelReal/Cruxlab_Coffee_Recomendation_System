import numpy as np
import pandas as pd
from typing import List, Dict

def calculate_ndcg(true_ratings: Dict[str, float], predicted_ids: List[str], n: int = 5) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain (NDCG) at rank N.
    
    Includes pre-processing to filter out NaN ratings and normalizes 
    relevance scores to a 0.0 - 1.0 range.
    """
    if not true_ratings or not predicted_ids:
        return 0.0

    # Data cleaning: Filter out ground truth items with missing ratings
    clean_true = {k: v for k, v in true_ratings.items() if not np.isnan(v)}
    if not clean_true:
        return 0.0

    # 1. Discounted Cumulative Gain (DCG) calculation
    # Normalizing ratings by 5.0 ensures relevance scale consistency
    dcg = 0.0
    for i, recipe_id in enumerate(predicted_ids[:n]):
        relevance = float(clean_true.get(recipe_id, 0.0)) / 5.0
        dcg += relevance / np.log2(i + 2)

    # 2. Ideal Discounted Cumulative Gain (IDCG) calculation
    # Represents the score if the model predicted the user's favorite items perfectly
    ideal_ratings = sorted(clean_true.values(), reverse=True)[:n]
    idcg = sum([
        (float(rel) / 5.0) / np.log2(idx + 2) 
        for idx, rel in enumerate(ideal_ratings)
    ])

    return dcg / idcg if idcg > 0 else 0.0

def get_user_relevance(user_id: str, recommended_ids: List[str], val_df: pd.DataFrame) -> List[float]:
    """
    Utility function to map a list of recommended IDs to their ground truth 
    relevance scores from the validation/test set.
    """
    user_validation_data = val_df[val_df['user_id'] == user_id]
    relevance_scores = []
    
    for recipe_id in recommended_ids:
        match = user_validation_data[user_validation_data['recipe_id'] == recipe_id]
        if not match.empty and not pd.isna(match.iloc[0]['rating']):
            relevance_scores.append(float(match.iloc[0]['rating']) / 5.0)
        else:
            relevance_scores.append(0.0)
            
    return relevance_scores