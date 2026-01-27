from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from recommender import recommend
from preprocess import load_all_data, preprocess_coffee_data
import uvicorn
import logging

# Configure logging for debugging and production monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Coffee Recommendation API")

# Global data containers
recipes = None
users = None
train = None

@app.on_event("startup")
def startup_event():
    """Load and preprocess data once when the server starts."""
    global recipes, users, train
    try:
        recipes_raw, users_raw, train_raw, _, _ = load_all_data()
        recipes, users, train = preprocess_coffee_data(recipes_raw, users_raw, train_raw)
        logger.info("Application data initialized successfully.")
    except Exception as e:
        logger.error(f"Startup failed: Could not load data files. Error: {e}")
        # Application will not start correctly without data
        raise RuntimeError("Data initialization failed. Check your CSV files and paths.")

@app.get("/")
async def root():
    """Health check endpoint to verify API status."""
    return JSONResponse(
        status_code=200, 
        content={"status": "online", "message": "Coffee Recommendation API is running"}
    )

@app.get("/recommend/{user_id}")
async def get_recommendations(user_id: str, n: int = 5):
    """
    Retrieve personalized coffee recommendations for a specific user.
    
    Args:
        user_id (str): Unique identifier of the user.
        n (int): Number of recommendations to return.
        
    Returns:
        dict: User ID and a list of recommended recipes with scores.
    """
    # 1. Validation: Check if requested number of items is positive
    if n <= 0:
        raise HTTPException(
            status_code=400, 
            detail="Parameter 'n' must be a positive integer greater than zero."
        )

    # 2. Validation: Check if user exists in the database
    if user_id not in users['user_id'].values:
        raise HTTPException(
            status_code=404, 
            detail=f"User ID '{user_id}' was not found in our records."
        )

    try:
        # 3. Core Logic: Generate recommendations using the hybrid model
        recommendations = recommend(user_id, users, recipes, train, n=n)
        
        if not recommendations:
            return {
                "user_id": user_id, 
                "recommendations": [], 
                "message": "No matching recipes found based on your equipment and tastes."
            }
        
        # 4. Data Transformation: Format output for the client
        result = []
        for r_id, score in recommendations:
            recipe_match = recipes[recipes['recipe_id'] == r_id]
            
            if recipe_match.empty:
                logger.warning(f"Recipe ID {r_id} recommended but not found in recipes database.")
                continue
                
            recipe_info = recipe_match.iloc[0]
            result.append({
                "recipe_id": r_id,
                "name": recipe_info['name'],
                "score": round(float(score), 4),
                "equipment": recipe_info['required_equipment']
            })
            
        return {"user_id": user_id, "recommendations": result}

    except Exception as e:
        # 5. Unexpected Internal Errors: Log and return 500
        logger.error(f"Internal error for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="An internal server error occurred while processing your recommendations."
        )

if __name__ == "__main__":
    # Start the application server locally
    uvicorn.run(app, host="127.0.0.1", port=8000)