import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocess import load_all_data, preprocess_coffee_data
from recommender import recommend, recommend_popular
from evaluation import calculate_ndcg

def evaluate(name: str, u: pd.DataFrame, r: pd.DataFrame, t: pd.DataFrame, test: pd.DataFrame) -> float:
    """Computes Mean NDCG@5 for a specific strategy."""
    scores = []
    for uid in test['user_id'].unique():
        true = dict(zip(test[test['user_id'] == uid]['recipe_id'], test[test['user_id'] == uid]['rating']))
        try:
            # Dispatching based on strategy name
            if name == "popularity":
                recs = recommend_popular(uid, u, r, t, n=5)
            else:
                recs = recommend(uid, u, r, t, n=5, strategy=name)
            
            preds = [rid for rid, _ in recs]
            scores.append(calculate_ndcg(true, preds, n=5))
        except: continue
    return np.mean(scores) if scores else 0.0

def main():
    r_r, u_r, t_r, test_raw, _ = load_all_data()
    test_raw = test_raw.dropna(subset=['rating'])
    recipes, users, train = preprocess_coffee_data(r_r, u_r, t_r)

    # Full experiment registry
    strats = {
        "hybrid_ml": "Hybrid LightGBM (SOTA)",
        "popularity": "Rule-Based (Popularity)",
        "weighted_content": "Weighted Content (Research)", # Новий метод
        "content_only": "Pure Content (Euclidean)",
        "user_hybrid": "User-User Hybrid (Cosine)"
    }
    
    results = []
    for key, label in strats.items():
        print(f"Benchmarking: {label}...")
        score = evaluate(key, users, recipes, train, test_raw)
        results.append({"Strategy": label, "NDCG@5": round(score, 4)})

    df = pd.DataFrame(results).sort_values("NDCG@5", ascending=False)
    print("\nFinal Benchmark Results:\n", df.to_string(index=False))

    # Visualization - Image only
    plt.figure(figsize=(12, 6))
    df_plot = df.sort_values("NDCG@5")
    plt.barh(df_plot['Strategy'], df_plot['NDCG@5'], color='#5da5da')
    plt.xlabel('NDCG@5 Score')
    plt.title('Performance Comparison: Global vs Personalized')
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig('strategy_comparison.png', dpi=300)
    print("\n[Success] Strategy comparison plot saved to file.")

if __name__ == "__main__":
    main()