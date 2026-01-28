import numpy as np
import itertools
from preprocess import load_all_data, preprocess_coffee_data
from evaluation import calculate_ndcg
from recommender import filter_by_equipment

def evaluate_weights(weights, users_df, recipes_df, test_df):
    """Calculates Mean NDCG for a specific set of weights."""
    scores = []
    
    # Process test users
    for uid in test_df['user_id'].unique():
        user_test = test_df[test_df['user_id'] == uid]
        if user_test.empty: continue
        
        true_ratings = dict(zip(user_test['recipe_id'], user_test['rating']))
        u_info = users_df[users_df['user_id'] == uid].iloc[0]
        avail = filter_by_equipment(u_info['owned_equipment'], recipes_df)
        
        if avail.empty: continue

        # Calculate weighted Euclidean distance
        diff_sum = 0
        for taste, w in weights.items():
            u_val = u_info['pref_strength_norm'] if taste == 'strength' else u_info[f'taste_pref_{taste}']
            r_val = avail['strength_norm'] if taste == 'strength' else avail[f'taste_{taste}']
            diff_sum += w * (r_val - u_val)**2
            
        avail['score'] = 1 / (1 + np.sqrt(diff_sum))
        preds = avail.sort_values('score', ascending=False).head(5)['recipe_id'].tolist()
        
        scores.append(calculate_ndcg(true_ratings, preds, n=5))
        
    return np.mean(scores) if scores else 0.0

def main():
    print("Loading data for weight optimization...")
    r_raw, u_raw, t_raw, test_raw, _ = load_all_data()
    test_raw = test_raw.dropna(subset=['rating'])
    recipes, users, train = preprocess_coffee_data(r_raw, u_raw, t_raw)

    # Define weight ranges to test
    # 1.0 is baseline, higher means more important
    search_space = [0.5, 1.0, 2.0] 
    tastes = ['bitterness', 'sweetness', 'acidity', 'body', 'strength']
    
    combinations = list(itertools.product(search_space, repeat=len(tastes)))
    print(f"Total combinations to test: {len(combinations)}")

    best_score = -1
    best_weights = None

    for i, combo in enumerate(combinations):
        current_weights = dict(zip(tastes, combo))
        score = evaluate_weights(current_weights, users, recipes, train, test_raw)
        
        if score > best_score:
            best_score = score
            best_weights = current_weights
            print(f"[{i+1}/{len(combinations)}] New best NDCG: {best_score:.4f} | Weights: {best_weights}")

    print("\n" + "="*30)
    print("OPTIMIZATION COMPLETE")
    print(f"Best NDCG: {best_score:.4f}")
    print(f"Optimal Weights: {best_weights}")
    print("="*30)

if __name__ == "__main__":
    main()