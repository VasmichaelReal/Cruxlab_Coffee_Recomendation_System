import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from recommender import recommend
from preprocess import load_all_data, preprocess_coffee_data

# Data state management
recipes, users, train = None, None, None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global recipes, users, train
    # Initialization phase
    r_raw, u_raw, t_raw, _, _ = load_all_data()
    recipes, users, train = preprocess_coffee_data(r_raw, u_raw, t_raw)
    yield

app = FastAPI(title="Coffee AI Coach API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    """Health check root endpoint."""
    return {"status": "online", "health": "ok", "version": "1.0.0"}

@app.get("/users")
async def get_all_users():
    """Returns list of users with cold start identifiers."""
    warm_ids = set(train['user_id'].unique())
    return {
        "users": [
            {"user_id": str(uid), "is_cold": uid not in warm_ids} 
            for uid in users['user_id']
        ]
    }

@app.get("/user/{user_id}")
async def get_profile(user_id: str):
    """Returns full user profile including all 5 taste parameters."""
    match = users[users['user_id'] == user_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="User not found")
    
    u = match.iloc[0]
    return {
        "user_id": user_id,
        "owned_equipment": list(u['owned_equipment']),
        "preferences": {
            "bitterness": float(u['taste_pref_bitterness']),
            "sweetness": float(u['taste_pref_sweetness']),
            "acidity": float(u['taste_pref_acidity']),
            "body": float(u['taste_pref_body']),
            "strength": float(u['pref_strength_norm']) # Strength enabled for UI
        }
    }

@app.get("/recommend/{user_id}")
async def get_rec(user_id: str, n: int = 5, strategy: str = "hybrid_ml"):
    """
    Main recommendation endpoint. Includes 'justification' to prevent UI errors.
    """
    if user_id not in users['user_id'].values:
        raise HTTPException(status_code=404, detail="Unknown User")

    try:
        is_cold = bool(train[train['user_id'] == user_id].empty)
        recs = recommend(user_id, users, recipes, train, n=n, strategy=strategy)
        
        output = []
        for r_id, score in recs:
            recipe = recipes[recipes['recipe_id'] == r_id].iloc[0]
            
            # Contextual justification for the UI
            justification = (
                "Matches your profile through flavor similarity (Cold Start)." if is_cold 
                else "Selected based on your positive brewing history (ML Ranking)."
            )

            output.append({
                "recipe_id": str(r_id),
                "name": str(recipe['name']),
                "score": round(float(score), 4),
                "details": {
                    "justification": justification, # Key required by app.py
                    "equipment": list(recipe['required_equipment']),
                    "flavor": {
                        "bitterness": float(recipe['taste_bitterness']),
                        "sweetness": float(recipe['taste_sweetness']),
                        "acidity": float(recipe['taste_acidity']),
                        "body": float(recipe['taste_body']),
                        "strength": float(recipe['strength_norm'])
                    }
                }
            })
        
        return {
            "user_id": user_id,
            "is_cold_start": is_cold,
            "recommendations": output
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)