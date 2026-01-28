def format_user_profile(user_row):
    """Parses raw user data into a structured profile response."""
    return {
        "user_id": user_row['user_id'],
        "owned_equipment": list(user_row['owned_equipment']),
        "preferences": {
            "bitterness": float(user_row['taste_pref_bitterness']),
            "sweetness": float(user_row['taste_pref_sweetness']),
            "acidity": float(user_row['taste_pref_acidity']),
            "body": float(user_row['taste_pref_body']),
            "strength": float(user_row['pref_strength_norm'])
        }
    }

def format_recommendation(recipe_row, score, is_cold):
    """Formats a single recipe recommendation with justification."""
    justification = (
        "Matches your profile through flavor similarity (Cold Start)." if is_cold 
        else "Selected based on your positive brewing history (ML Ranking)."
    )
    return {
        "recipe_id": str(recipe_row['recipe_id']),
        "name": str(recipe_row['name']),
        "score": round(float(score), 4),
        "details": {
            "justification": justification, 
            "equipment": list(recipe_row['required_equipment']),
            "flavor": {
                "bitterness": float(recipe_row['taste_bitterness']),
                "sweetness": float(recipe_row['taste_sweetness']),
                "acidity": float(recipe_row['taste_acidity']),
                "body": float(recipe_row['taste_body']),
                "strength": float(recipe_row['strength_norm'])
            }
        }
    }