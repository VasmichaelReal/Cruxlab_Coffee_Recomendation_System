import os
from datetime import datetime
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation import calculate_ndcg
from preprocess import load_all_data, preprocess_coffee_data
from recommender import recommend, recommend_popular

# Absolute path for modular project structure
DS_DIR = os.path.dirname(os.path.abspath(__file__))

def evaluate(
    strategy_name: str, 
    users: pd.DataFrame, 
    recipes: pd.DataFrame, 
    train: pd.DataFrame, 
    test: pd.DataFrame
) -> float:
    """
    Computes Mean NDCG@5 for a given recommendation strategy using per-user macro-averaging.
    Optimizes global popularity calculations to reduce redundant processing.
    """
    scores = []
    
    # Pre-calculate global popularity scores once for the entire strategy evaluation
    pre_calculated_popularity = None
    if strategy_name == "popularity":
        print(f"  Optimizing: Pre-calculating global temporal weights for {strategy_name}...")
        temp_train = train.copy()
        temp_train['timestamp'] = pd.to_datetime(temp_train['timestamp'])
        now = datetime.now()
        
        # Consistent temporal decay logic: Lambda=0.005, Floor=0.1
        temp_train['days_diff'] = (now - temp_train['timestamp']).dt.days
        temp_train['score'] = np.maximum(np.exp(-0.005 * temp_train['days_diff']), 0.1)
        pre_calculated_popularity = temp_train.groupby('recipe_id')['score'].sum().reset_index()

    # Iterate over unique users in the test set to evaluate personalization
    for user_id in test['user_id'].unique():
        user_test_data = test[test['user_id'] == user_id]
        true_ratings = dict(zip(user_test_data['recipe_id'], user_test_data['rating']))
        
        try:
            if strategy_name == "popularity":
                recommendations = recommend_popular(
                    user_id, users, recipes, train, n=5, 
                    precalculated_pop=pre_calculated_popularity
                )
            else:
                recommendations = recommend(user_id, users, recipes, train, n=5, strategy=strategy_name)
            
            predicted_ids = [recipe_id for recipe_id, _ in recommendations]
            scores.append(calculate_ndcg(true_ratings, predicted_ids, n=5))
        except Exception:
            continue
        
    return np.mean(scores) if scores else 0.0

def main():
    """
    Executes a comprehensive benchmark of recommendation strategies and 
    generates an analytical performance visualization.
    """
    # Load and preprocess datasets using modular components
    raw_recipes, raw_users, raw_train, raw_test, _ = load_all_data()
    raw_test = raw_test.dropna(subset=['rating'])
    recipes, users, train = preprocess_coffee_data(raw_recipes, raw_users, raw_train)

    # Strategy definitions and human-readable labels
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

    # Sort results for logical visualization
    df_results = pd.DataFrame(results).sort_values("NDCG@5", ascending=True)
    print("\nBenchmark Summary:\n", df_results.sort_values("NDCG@5", ascending=False).to_string(index=False))

    # Analytical Visualization
    plt.figure(figsize=(13, 7))
    
    # Categorize strategies by architectural approach
    strategy_mapping = {
        "Hybrid LightGBM (SOTA)": "Personalized (Dynamic/ML)",
        "Weighted Content (Research)": "Personalized (Dynamic/ML)",
        "User-User Hybrid (Cosine)": "Personalized (Dynamic/ML)",
        "Time-Aware Popularity": "Global Baseline (Time-Aware)",
        "Pure Content (Euclidean)": "Global Baseline (Time-Aware)"
    }
    
    color_scheme = {
        "Personalized (Dynamic/ML)":"#3498db",
        "Global Baseline (Time-Aware)":"#95a5a6"
    }

    processed_categories = set()
    for _, row in df_results.iterrows():
        category = strategy_mapping.get(row['Strategy'])
        label = category if category not in processed_categories else ""
        
        plt.barh(
            row['Strategy'], 
            row['NDCG@5'], 
            color=color_scheme[category], 
            label=label
        )
        processed_categories.add(category)

    # Plot Styling and Labeling
    plt.xlabel('Mean NDCG@5 (Average Per-User Ranking Accuracy)', fontsize=10, fontweight='bold')
    plt.title('Performance Comparison: Recommendation Strategies', fontsize=13, pad=15)
    plt.xlim(0, 1.1) 
    plt.grid(axis='x', linestyle='--', alpha=0.3)

    # Metric annotations on bars
    for idx, value in enumerate(df_results['NDCG@5']):
        plt.text(value + 0.01, idx, f'{value:.4f}', va='center', fontweight='bold')

    plt.legend(title="Algorithm Taxonomy", loc='lower right', frameon=True)
    plt.tight_layout()
    
    # Export visualization to data_science directory
    export_path = os.path.join(DS_DIR, 'strategy_comparison.png')
    plt.savefig(export_path, dpi=300, bbox_inches='tight')
    print(f"\nReport Generated: Comparison chart saved to {export_path}")

if __name__ == "__main__":
    main()