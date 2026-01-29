import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from typing import List, Dict, Any

# --- 1. CONFIGURATION & LOADING ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'coffee_model_optimized.pkl')

model = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ ML Model loaded: {MODEL_PATH}")
    else:
        print(f"⚠️ ML Model not found at {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# --- 2. HELPER FUNCTIONS ---

def filter_by_equipment(user_owned_equipment, recipes_df):
    """Filters recipes based on user equipment."""
    if isinstance(user_owned_equipment, str):
        import ast
        try:
            user_set = set(ast.literal_eval(user_owned_equipment))
        except:
            user_set = {user_owned_equipment}
    else:
        user_set = set(user_owned_equipment)
        
    eq_col = 'equipment' if 'equipment' in recipes_df.columns else 'required_equipment'
    if eq_col not in recipes_df.columns:
        return recipes_df.copy()

    def check(req):
        if not req: return True
        if isinstance(req, (list, tuple, set, np.ndarray)):
            return set(req).issubset(user_set)
        return req in user_set

    mask = recipes_df[eq_col].apply(check)
    return recipes_df[mask].copy()

def prepare_features_for_ml(user_row, candidates_df):
    """
    Constructs features EXACTLY as the optimized LightGBM expects them.
    """
    df = candidates_df.copy()
    u = user_row.to_dict()
    
    # 1. User Preferences Mapping
    def get_pref(key_base):
        for k in [f'taste_pref_{key_base}', f'pref_{key_base}', f'preferred_{key_base}']:
            if k in u and pd.notnull(u[k]): return float(u[k])
        return 0.5

    df['taste_pref_bitterness'] = get_pref('bitterness')
    df['taste_pref_sweetness'] = get_pref('sweetness')
    df['taste_pref_acidity'] = get_pref('acidity')
    df['taste_pref_body'] = get_pref('body')
    df['pref_strength_norm'] = get_pref('strength')

    # 2. Recipe Tastes
    tastes = ['bitterness', 'sweetness', 'acidity', 'body']
    for t in tastes:
        if f'taste_{t}' not in df.columns and t in df.columns:
            df[f'taste_{t}'] = df[t]

    # 3. Strength Normalization
    if 'strength_norm' not in df.columns:
        df['strength_norm'] = df['strength'] / 10.0 if 'strength' in df.columns else 0.5

    # 4. Deltas
    for t in tastes:
        df[f'delta_{t}'] = abs(df[f'taste_{t}'] - df[f'taste_pref_{t}'])
    
    df['strength_match'] = (abs(df['strength_norm'] - df['pref_strength_norm']) < 0.2).astype(int)

    features = [
        'taste_bitterness', 'taste_sweetness', 'taste_acidity', 'taste_body', 'strength_norm',
        'taste_pref_bitterness', 'taste_pref_sweetness', 'taste_pref_acidity', 'taste_pref_body', 
        'pref_strength_norm', 'delta_bitterness', 'delta_sweetness', 'delta_acidity', 'delta_body', 'strength_match'
    ]
    
    for f in features:
        if f not in df.columns: df[f] = 0.0
        
    return df[features]

def generate_recommendation_details(user_info: pd.Series, recipe: pd.Series) -> Dict[str, Any]:
    """Generates justification text and flavor profile for the UI."""
    tastes = ['bitterness', 'sweetness', 'acidity', 'body']
    
    # Safe getters
    def get_u_pref(t):
        return float(user_info.get(f'taste_pref_{t}', user_info.get(f'pref_{t}', 0.5)))
    
    def get_r_val(t):
        return float(recipe.get(f'taste_{t}', recipe.get(t, 0.5)))

    deltas = {t: abs(get_r_val(t) - get_u_pref(t)) for t in tastes}
    
    # Justification logic
    if deltas['body'] < 0.15:
        justification = "Matches your preference for body and texture."
    elif deltas['sweetness'] < 0.15:
        justification = "Excellent match for your sweetness level."
    elif deltas['acidity'] < 0.15:
        justification = "Aligns well with your acidity preference."
    else:
        best_match = min(deltas, key=deltas.get)
        justification = f"Selected based on your taste profile ({best_match})."

    return {
        "justification": justification,
        "flavor": {
            "bitterness": get_r_val('bitterness'),
            "sweetness": get_r_val('sweetness'),
            "acidity": get_r_val('acidity'),
            "body": get_r_val('body'),
            "strength": float(recipe.get('strength', 5))
        },
        "equipment": recipe.get('equipment', [])
    }

# --- 3. STRATEGIES ---

def recommend_ml(user_id, users, recipes, train, n=5):
    if model is None:
        return recommend_content(user_id, users, recipes, train, n)

    user_info = users[users['user_id'] == user_id].iloc[0]
    available = filter_by_equipment(user_info['owned_equipment'], recipes)
    
    if available.empty: return []

    try:
        X_pred = prepare_features_for_ml(user_info, available)
        probs = model.predict_proba(X_pred)[:, 1]
        
        available = available.copy()
        available['score'] = probs
        # Return ID and Score for enrichment
        return available.sort_values('score', ascending=False).head(n)[['recipe_id', 'score']].to_dict('records')
    except Exception as e:
        print(f"ML Error: {e}")
        return recommend_content(user_id, users, recipes, train, n)

def recommend_popular(user_id, users, recipes, train, n=5, precalculated_pop=None):
    user_info = users[users['user_id'] == user_id].iloc[0]
    available = filter_by_equipment(user_info['owned_equipment'], recipes)
    
    if available.empty: return []

    if precalculated_pop is not None:
        res = available.merge(precalculated_pop, on='recipe_id', how='left').fillna(0)
    else:
        temp_train = train.copy()
        temp_train['timestamp'] = pd.to_datetime(temp_train['timestamp'])
        days_diff = (datetime.now() - temp_train['timestamp']).dt.days
        temp_train['w'] = np.maximum(np.exp(-0.005 * days_diff), 0.1)
        pop = temp_train.groupby('recipe_id')['w'].sum().reset_index().rename(columns={'w': 'score'})
        res = available.merge(pop, on='recipe_id', how='left').fillna(0)
    
    return res.sort_values('score', ascending=False).head(n)[['recipe_id', 'score']].to_dict('records')

def recommend_content(user_id, users, recipes, train, n=5, weights=None):
    user_info = users[users['user_id'] == user_id].iloc[0]
    available = filter_by_equipment(user_info['owned_equipment'], recipes)
    
    if available.empty: return []

    # Use ML features for robust calculation
    feat = prepare_features_for_ml(user_info, available)
    
    if weights is None:
        weights = {'delta_bitterness': 1.0, 'delta_sweetness': 1.0, 'delta_acidity': 1.0, 'delta_body': 1.0}

    dist = 0
    for col, w in weights.items():
        if col in feat.columns:
            dist += w * (feat[col] ** 2)
            
    available = available.copy()
    available['score'] = 1 / (1 + np.sqrt(dist))
    
    return available.sort_values('score', ascending=False).head(n)[['recipe_id', 'score']].to_dict('records')

# --- 4. MAIN DISPATCHER (ENRICHMENT LAYER) ---

def recommend(user_id, users, recipes, train, n=5, strategy='hybrid_ml'):
    """
    Returns full recommendation objects (Name, Score, Details) for the Frontend.
    """
    try:
        # 1. Get Raw Recommendations (ID + Score)
        raw_recs = []
        if strategy == 'hybrid_ml':
            raw_recs = recommend_ml(user_id, users, recipes, train, n)
        elif strategy == 'popularity':
            raw_recs = recommend_popular(user_id, users, recipes, train, n)
        elif strategy == 'weighted_content':
            w = {'delta_bitterness': 1.2, 'delta_sweetness': 1.5, 'delta_acidity': 1.0, 'delta_body': 0.8}
            raw_recs = recommend_content(user_id, users, recipes, train, n, weights=w)
        else:
            raw_recs = recommend_content(user_id, users, recipes, train, n)

        # 2. Enrich with Name, Details, Justification
        final_output = []
        user_info = users[users['user_id'] == user_id].iloc[0]

        for item in raw_recs:
            rid = item['recipe_id']
            score = item.get('score', 0.0)
            
            # Find recipe row
            recipe_row = recipes[recipes['recipe_id'] == rid]
            if recipe_row.empty: continue
            recipe_row = recipe_row.iloc[0]
            
            # Build full object for frontend
            rec_obj = {
                "recipe_id": str(rid),
                "name": str(recipe_row.get('name', 'Unknown Coffee')),
                "score": round(float(score), 2),
                "details": generate_recommendation_details(user_info, recipe_row)
            }
            final_output.append(rec_obj)
            
        return final_output

    except Exception as e:
        print(f"Global Rec Error: {e}")
        return []