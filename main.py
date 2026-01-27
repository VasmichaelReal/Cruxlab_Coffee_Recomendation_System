import uvicorn
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from recommender import recommend
from preprocess import load_all_data, preprocess_coffee_data

# 1. Logging configuration for production-ready monitoring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Global data storage
recipes, users, train = None, None, None

# 2. Modern Lifespan handler to manage data loading
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for the API."""
    global recipes, users, train
    logger.info("Starting up: Loading coffee recommendation engine...")
    try:
        # Load and preprocess data only once at startup
        r_raw, u_raw, t_raw, _, _ = load_all_data()
        recipes, users, train = preprocess_coffee_data(r_raw, u_raw, t_raw)
        logger.info("System Ready: All datasets loaded successfully.")
        yield
    except Exception as e:
        logger.critical(f"Startup failed: {str(e)}")
        raise RuntimeError(f"Application could not start due to: {e}")
    finally:
        logger.info("Shutting down: Cleaning up resources...")

app = FastAPI(
    title="Coffee AI Coach API",
    description="ML-powered personalized coffee recommendation engine.",
    lifespan=lifespan
)

# 3. CORS configuration for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    """Simple status check for the API service."""
    return {"status": "online", "engine": "Hybrid LightGBM/Euclidean"}

@app.get("/users")
async def get_all_users():
    """Returns the list of all users with cold/warm flags for UI."""
    if users is None:
        raise HTTPException(status_code=503, detail="System not initialized.")
    
    warm_users = set(train['user_id'].unique())
    user_list = []
    for _, row in users.iterrows():
        u_id = str(row['user_id'])
        user_list.append({
            "user_id": u_id,
            "is_cold": bool(u_id not in warm_users)  # Native bool for JSON
        })
    return {"users": user_list}

@app.get("/user/{user_id}")
async def get_user_profile(user_id: str):
    """Returns specific user taste preferences and equipment."""
    user_match = users[users['user_id'] == user_id]
    if user_match.empty:
        raise HTTPException(status_code=404, detail="User not found.")
    
    u = user_match.iloc[0]
    return {
        "user_id": str(u['user_id']),
        "owned_equipment": list(u['owned_equipment']),
        "preferences": {
            "bitterness": float(u['taste_pref_bitterness']),
            "sweetness": float(u['taste_pref_sweetness']),
            "acidity": float(u['taste_pref_acidity']),
            "body": float(u['taste_pref_body']),
            "strength": int(u['preferred_strength'])  # JSON-safe int
        }
    }

@app.get("/recommend/{user_id}")
async def get_recommendations(user_id: str, n: int = 5):
    """
    Core logic: Dispatches between LightGBM (Warm) and Euclidean (Cold).
    """
    if users is None or train is None:
        raise HTTPException(status_code=503, detail="System not initialized.")

    if user_id not in users['user_id'].values:
        raise HTTPException(status_code=404, detail="User ID not found.")

    try:
        # Determine user state
        is_cold = bool(train[train['user_id'] == user_id].empty)
        
        # Call the optimized recommend function from recommender.py
        # It automatically uses the best strategy based on is_cold status
        recs = recommend(user_id, users, recipes, train, n=n)

        result = []
        for r_id, score in recs:
            recipe_info = recipes[recipes['recipe_id'] == r_id].iloc[0]
            
            # Explainable AI (XAI) Justification
            justification = (
                "Matches your flavor profile through direct taste similarity (Cold Start)." if is_cold 
                else "Based on patterns in your successful brewing history (ML Ranking)."
            )

            result.append({
                "recipe_id": str(r_id),
                "name": str(recipe_info['name']),
                "score": round(float(score), 4),
                "details": {
                    "equipment": list(recipe_info['required_equipment']),
                    "strength": int(recipe_info['strength']),
                    "flavor": {
                        "bitterness": float(recipe_info['taste_bitterness']),
                        "sweetness": float(recipe_info['taste_sweetness']),
                        "acidity": float(recipe_info['taste_acidity']),
                        "body": float(recipe_info['taste_body'])
                    },
                    "justification": justification
                }
            })
            
        return {
            "user_id": user_id, 
            "is_cold_start": is_cold,
            "strategy_used": "Content-Based" if is_cold else "Hybrid-LightGBM",
            "recommendations": result
        }

    except Exception as e:
        logger.error(f"Error generating recommendation for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Recommendation engine error.")

if __name__ == "__main__":
    # Run with auto-reload for development
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)