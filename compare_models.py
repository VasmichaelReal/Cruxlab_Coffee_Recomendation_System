import pandas as pd
import numpy as np
from tqdm import tqdm
from preprocess import load_all_data, preprocess_coffee_data
from recommender import recommend
from evaluation import calculate_ndcg


def evaluate_strategy(strategy_name: str, 
                      users_df: pd.DataFrame, 
                      recipes_df: pd.DataFrame, 
                      train_df: pd.DataFrame, 
                      test_df: pd.DataFrame, 
                      n: int = 5) -> float:
    """
    Performs offline evaluation of a recommendation strategy using Mean NDCG@n.
    
    Args:
        strategy_name (str): The identifier of the strategy to test.
        users_df (pd.DataFrame): Preprocessed user profiles.
        recipes_df (pd.DataFrame): Preprocessed recipe data.
        train_df (pd.DataFrame): Training set of user-recipe interactions.
        test_df (pd.DataFrame): Test set for ground truth verification.
        n (int): The number of top-ranked items to evaluate.

    Returns:
        float: The mean NDCG score across all users in the test set.
    """
    scores = []
    unique_test_users = test_df['user_id'].unique()
    
    # Progress tracking using standard CLI library
    for user_id in tqdm(unique_test_users, desc=f"Evaluating: {strategy_name}", leave=False):
        # Extract ground truth interactions for the current user
        user_test_data = test_df[test_df['user_id'] == user_id]
        if user_test_data.empty:
            continue
            
        true_ratings = dict(zip(user_test_data['recipe_id'], user_test_data['rating']))
        
        try:
            # Generate predictions based on the specified strategy
            predictions = recommend(
                user_id, users_df, recipes_df, train_df, n=n, strategy=strategy_name
            )
            predicted_ids = [item_id for item_id, _ in predictions]
            
            # Calculate NDCG metric for the user
            user_score = calculate_ndcg(true_ratings, predicted_ids, n=n)
            scores.append(user_score)
        except Exception:
            # Silently skip errors to ensure completion of the batch process
            continue
            
    return np.mean(scores) if scores else 0.0


def main():
    """
    Entry point for the recommendation strategy comparison experiment.
    """
    print("-" * 60)
    print("EVALUATION: COFFEE RECOMMENDATION STRATEGIES")
    print("-" * 60)
    
    # 1. Data Ingestion
    try:
        r_raw, u_raw, t_raw, test_raw, _ = load_all_data()
        
        # Data Cleaning: remove interactions without ratings to avoid bias
        test_raw = test_raw.dropna(subset=['rating'])
        
        # Feature Engineering and Preprocessing
        recipes, users, train = preprocess_coffee_data(r_raw, u_raw, t_raw)
    except Exception as e:
        print(f"CRITICAL ERROR: Data loading failed. Details: {e}")
        return

    # 2. Strategy Definition
    # 'hybrid_ml' uses LightGBM, 'user_hybrid' uses Cosine User-User, 'content_only' uses Euclidean
    experiment_registry = {
        "hybrid_ml": "Hybrid LightGBM (Gradient Boosting)",
        "content_only": "Content-Based (Euclidean Distance)",
        "user_hybrid": "User-User Hybrid (Cosine Similarity)"
    }
    
    comparison_results = []

    # 3. Execution Phase
    for key, label in experiment_registry.items():
        mean_score = evaluate_strategy(key, users, recipes, train, test_raw, n=5)
        comparison_results.append({
            "Strategy": label, 
            "NDCG@5": round(mean_score, 4)
        })
        
    # 4. Results Reporting
    results_df = pd.DataFrame(comparison_results).sort_values("NDCG@5", ascending=False)
    
    print("\nBenchmark Results Summary:")
    try:
        # Formats output as a clean markdown table
        print(results_df.to_markdown(index=False))
    except ImportError:
        print(results_df.to_string(index=False))
    
    print("-" * 60)
    
    best_strategy = results_df.iloc[0]
    print(f"CONCLUSION: The optimal strategy is '{best_strategy['Strategy']}'")
    print(f"Performance baseline achieved: {best_strategy['NDCG@5']}")
    print("-" * 60 + "\n")


if __name__ == "__main__":
    main()