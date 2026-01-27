import numpy as np
import pandas as pd
from typing import List


def ndcg_at_k(relevances: List[float], k: int = 5) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at rank K.
    
    NDCG accounts for the position of relevant items: higher rank results 
    in a higher score based on the formula: rel_i / log2(i + 1).
    """
    relevances = np.asfarray(relevances)[:k]
    if relevances.size == 0:
        return 0.0

    # Calculate Discounted Cumulative Gain (DCG)
    # Positions are i+2 because enumerate starts at 0 (log2(1) is 0)
    dcg = np.sum(relevances / np.log2(np.arange(2, relevances.size + 2)))

    # Calculate Ideal Discounted Cumulative Gain (IDCG)
    idcg_relevances = sorted(relevances, reverse=True)
    idcg = np.sum(idcg_relevances / np.log2(np.arange(2, len(idcg_relevances) + 2)))

    return dcg / idcg if idcg > 0 else 0.0


def get_user_relevance(user_id: str, recommended_ids: List[str], val_df: pd.DataFrame) -> List[float]:
    """
    Map recommended IDs to ground truth relevance scores.
    
    Relevance is calculated as rating / 5.0. Non-interacted items 
    get a score of 0.0.
    """
    user_val = val_df[val_df['user_id'] == user_id]
    relevances = []
    
    for rid in recommended_ids:
        # Check if the user has interacted with this recipe in validation set
        match = user_val[user_val['recipe_id'] == rid]
        
        if not match.empty and not pd.isna(match.iloc[0]['rating']):
            # Normalized relevance score
            relevances.append(float(match.iloc[0]['rating']) / 5.0)
        else:
            relevances.append(0.0)
            
    return relevances