from fastapi import APIRouter, HTTPException
import server.state as state
import server.services as services
from data_science.recommender import recommend
import traceback
import math

# Router object must be declared at the module level for main.py to import it
router = APIRouter()

@router.get("/")
async def health_check():
    """Service health and version status."""
    return {"status": "online", "health": "ok", "version": "1.0.0"}

@router.get("/users")
async def get_users():
    """Returns a list of all users with cold-start flags."""
    warm_ids = set(state.train['user_id'].unique())
    return {
        "users": [
            {"user_id": str(uid), "is_cold": uid not in warm_ids} 
            for uid in state.users['user_id']
        ]
    }

@router.get("/user/{user_id}")
async def get_user(user_id: str):
    """Fetches formatted user profile for the UI sidebar."""
    match = state.users[state.users['user_id'] == user_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="User not found")
    return services.format_user_profile(match.iloc[0])

@router.get("/recommend/{user_id}")
async def get_rec(user_id: str, n: int = 5, strategy: str = "hybrid_ml"):
    """
    Core recommendation endpoint.
    Retrieves actual ratings from the validation set (state.val).
    """
    if user_id not in state.users['user_id'].values:
        raise HTTPException(status_code=404, detail="Unknown User")

    try:
        # Check if user has no interaction history
        is_cold = state.train[state.train['user_id'] == user_id].empty
        
        # Call recommender passing the validation set for actual rating lookup
        ml_results = recommend(
            user_id, 
            state.users, 
            state.recipes, 
            state.train, 
            state.val, 
            n=n, 
            strategy=strategy
        )
        
        # Defensive fix: Filter out any items where the score is NaN to prevent JSON crash
        # This occurs if a rating is missing in the source CSV/dataframe
        valid_results = [
            res for res in ml_results 
            if not (isinstance(res.get('score'), float) and math.isnan(res.get('score')))
        ]
        
        # Process results and round scores
        formatted_recs = services.process_ml_results(valid_results, is_cold)
        
        return {
            "user_id": user_id, 
            "is_cold_start": bool(is_cold), 
            "recommendations": formatted_recs
        }
    except Exception as e:
        # Print full error log to the server terminal for debugging
        print("--- SERVER ERROR TRACEBACK ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Recommendation Engine Error: {str(e)}")