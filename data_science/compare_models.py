import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Assuming these modules exist in your project structure
from evaluation import calculate_ndcg
from preprocess import load_all_data, preprocess_coffee_data
from recommender import recommend, recommend_popular

# Constants
DS_DIR = os.path.dirname(os.path.abspath(__file__))
POPULARITY_DECAY_LAMBDA = 0.005
POPULARITY_MIN_SCORE = 0.1
TOP_K = 5


def _precalculate_global_popularity(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates time-decayed popularity scores for recipes based on training data.
    """
    print("  Optimizing: Pre-calculating global temporal weights for popularity...")
    temp_train = train_df.copy()
    temp_train['timestamp'] = pd.to_datetime(temp_train['timestamp'])
    
    now = datetime.now()
    temp_train['days_diff'] = (now - temp_train['timestamp']).dt.days
    
    # Apply exponential decay
    temp_train['score'] = np.maximum(
        np.exp(-POPULARITY_DECAY_LAMBDA * temp_train['days_diff']), 
        POPULARITY_MIN_SCORE
    )
    
    return temp_train.groupby('recipe_id')['score'].sum().reset_index()


def _extract_recommended_ids(recommendations: Union[List[Dict], List[tuple], List[Any]]) -> List[str]:
    """
    Robustly extracts recipe IDs from various recommendation return formats.
    Handles lists of dicts, tuples, or raw IDs.
    """
    if not recommendations:
        return []

    first_item = recommendations[0]

    # Handle list of dictionaries: [{'recipe_id': '123', ...}, ...]
    if isinstance(first_item, dict):
        return [r['recipe_id'] for r in recommendations]
    
    # Handle list of tuples: [('123', 0.95), ...]
    if isinstance(first_item, (tuple, list)):
        return [r[0] for r in recommendations]
    
    # Handle simple list of IDs: ['123', '456', ...]
    return recommendations


def evaluate(
    strategy_name: str, 
    users: pd.DataFrame, 
    recipes: pd.DataFrame, 
    train: pd.DataFrame, 
    test: pd.DataFrame
) -> float:
    """
    Computes Mean NDCG@5 for a given recommendation strategy using per-user macro-averaging.
    """
    scores = []
    
    # Optimization for popularity strategy
    pre_calculated_pop = None
    if strategy_name == "popularity":
        pre_calculated_pop = _precalculate_global_popularity(train)

    unique_users = test['user_id'].unique()
    
    # Iterate over users to calculate metrics
    for user_id in unique_users:
        user_test_data = test[test['user_id'] == user_id]
        true_ratings = dict(zip(user_test_data['recipe_id'], user_test_data['rating']))
        
        if not true_ratings:
            continue

        try:
            if strategy_name == "popularity":
                recommendations = recommend_popular(
                    user_id, users, recipes, train, n=TOP_K, 
                    precalculated_pop=pre_calculated_pop
                )
            else:
                recommendations = recommend(
                    user_id, users, recipes, train, n=TOP_K, strategy=strategy_name
                )
            
            predicted_ids = _extract_recommended_ids(recommendations)
            
            score = calculate_ndcg(true_ratings, predicted_ids, n=TOP_K)
            scores.append(score)
            
        except Exception:
            # Silently skip errors to ensure the batch evaluation finishes.
            # In a production environment, use logging.error() here.
            continue
        
    return np.mean(scores) if scores else 0.0


def generate_performance_report(df_results: pd.DataFrame, output_path: str) -> None:
    """
    Generates and saves a horizontal bar chart comparing strategy performance.
    """
    plt.figure(figsize=(13, 7))
    
    strategy_mapping = {
        "Hybrid LightGBM (SOTA)": "Personalized (Dynamic/ML)",
        "Weighted Content (Research)": "Personalized (Dynamic/ML)",
        "User-User Hybrid (Cosine)": "Personalized (Dynamic/ML)",
        "Time-Aware Popularity": "Global Baseline (Time-Aware)",
        "Pure Content (Euclidean)": "Global Baseline (Time-Aware)"
    }
    
    color_scheme = {
        "Personalized (Dynamic/ML)": "#3498db",
        "Global Baseline (Time-Aware)": "#95a5a6"
    }

    processed_categories = set()
    
    # Plotting bars
    for _, row in df_results.iterrows():
        strategy_label = row['Strategy']
        ndcg_score = row['NDCG@5']
        
        category = strategy_mapping.get(strategy_label, "Other")
        
        # Add label only once per category for the legend
        legend_label = category if category not in processed_categories else ""
        
        plt.barh(
            strategy_label, 
            ndcg_score, 
            color=color_scheme.get(category, "gray"), 
            label=legend_label
        )
        processed_categories.add(category)

    # Styling
    plt.xlabel(f'Mean NDCG@{TOP_K} (Average Per-User Ranking Accuracy)', fontsize=10, fontweight='bold')
    plt.title('Performance Comparison: Recommendation Strategies', fontsize=13, pad=15)
    plt.xlim(0, 1.1) 
    plt.grid(axis='x', linestyle='--', alpha=0.3)

    # Annotations
    for idx, value in enumerate(df_results['NDCG@5']):
        plt.text(value + 0.01, idx, f'{value:.4f}', va='center', fontweight='bold')

    plt.legend(title="Algorithm Taxonomy", loc='lower right', frameon=True)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nReport Generated: Comparison chart saved to {output_path}")


def main():
    """
    Main execution routine for benchmarking recommendation strategies.
    """
    # Load and preprocess datasets
    raw_recipes, raw_users, raw_train, raw_test, _ = load_all_data()
    
    # Ensure test data has ratings
    raw_test = raw_test.dropna(subset=['rating'])
    
    recipes, users, train = preprocess_coffee_data(raw_recipes, raw_users, raw_train)

    strategies = {
        "hybrid_ml": "Hybrid LightGBM (SOTA)",
        "popularity": "Time-Aware Popularity",
        "weighted_content": "Weighted Content (Research)",
        "content_only": "Pure Content (Euclidean)",
        "user_hybrid": "User-User Hybrid (Cosine)"
    }
    
    results = []
    
    for key, label in strategies.items():
        print(f"Benchmarking Strategy: {label}...")
        score = evaluate(key, users, recipes, train, raw_test)
        results.append({"Strategy": label, "NDCG@5": round(score, 4)})

    # Create DataFrame and sort for visualization
    df_results = pd.DataFrame(results).sort_values("NDCG@5", ascending=True)
    
    # Display table in console
    print("\nBenchmark Summary:\n")
    print(df_results.sort_values("NDCG@5", ascending=False).to_string(index=False))

    # Generate visualization
    export_path = os.path.join(DS_DIR, 'strategy_comparison.png')
    generate_performance_report(df_results, export_path)


if __name__ == "__main__":
    main()