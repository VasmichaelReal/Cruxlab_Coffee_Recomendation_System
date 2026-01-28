from fastapi import APIRouter, HTTPException
import server.state as state
import server.services as services
from data_science.recommender import recommend

router = APIRouter()

@router.get("/")
async def health_check():
    return {"status": "online", "health": "ok", "version": "1.0.0"}

@router.get("/users")
async def get_users():
    warm_ids = set(state.train['user_id'].unique())
    return {
        "users": [{"user_id": str(uid), "is_cold": uid not in warm_ids} 
                 for uid in state.users['user_id']]
    }

@router.get("/user/{user_id}")
async def get_user(user_id: str):
    match = state.users[state.users['user_id'] == user_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="User not found")
    return services.format_user_profile(match.iloc[0])

@router.get("/recommend/{user_id}")
async def get_rec(user_id: str, n: int = 5, strategy: str = "hybrid_ml"):
    if user_id not in state.users['user_id'].values:
        raise HTTPException(status_code=404, detail="Unknown User")

    try:
        is_cold = state.train[state.train['user_id'] == user_id].empty
        recs = recommend(user_id, state.users, state.recipes, state.train, n=n, strategy=strategy)
        
        results = []
        for r_id, score in recs:
            recipe_row = state.recipes[state.recipes['recipe_id'] == r_id].iloc[0]
            results.append(services.format_recommendation(recipe_row, score, is_cold))
        
        return {"user_id": user_id, "is_cold_start": bool(is_cold), "recommendations": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))