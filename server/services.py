def format_user_profile(user_row):
    """Formats the user profile with native types. Strength is normalized (0-1)."""
    return {
        "user_id": str(user_row['user_id']),
        "owned_equipment": [str(e) for e in user_row['owned_equipment']],
        "preferences": {
            "bitterness": float(user_row['taste_pref_bitterness']),
            "sweetness": float(user_row['taste_pref_sweetness']),
            "acidity": float(user_row['taste_pref_acidity']),
            "body": float(user_row['taste_pref_body']),
            "strength": float(user_row['pref_strength_norm'])
        }
    }

def process_ml_results(ml_recs: list, is_cold: bool):
    """Wraps ML results with cold-start flag."""
    for rec in ml_recs:
        rec['is_cold_start'] = bool(is_cold)
    return ml_recs