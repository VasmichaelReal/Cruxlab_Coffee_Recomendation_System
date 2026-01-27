import os
import pandas as pd

# Path to the data directory
DATA_PATH = 'data/'


def analyze_dataset(file_name):
    """
    Perform initial exploratory data analysis on a given CSV file.
    """
    file_path = os.path.join(DATA_PATH, file_name)
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    print(f"\n{'='*10} Analysis: {file_name} {'='*10}")
    df = pd.read_csv(file_path)
    
    # Preview top rows
    print("\n[Head]:")
    print(df.head(3))
    
    # Check for missing values
    print("\n[Missing Values]:")
    print(df.isnull().sum())
    
    # Statistics for flavor profiles
    taste_cols = ['bitterness', 'sweetness', 'acidity', 'body']
    pref_cols = ['taste_pref_bitterness', 'taste_pref_sweetness', 'taste_pref_acidity', 'taste_pref_body']

    # Check for recipe taste profiles
    if all(col in df.columns for col in taste_cols):
        print("\n[Recipe Taste Profile Statistics]:")
        print(df[taste_cols].describe())
        
    # Check for user preference profiles
    elif all(col in df.columns for col in pref_cols):
        print("\n[User Preference Statistics]:")
        print(df[pref_cols].describe())


if __name__ == "__main__":
    # Core datasets for the recommendation system
    target_files = ['recipes.csv', 'users.csv', 'interactions_train.csv']
    
    for file in target_files:
        analyze_dataset(file)