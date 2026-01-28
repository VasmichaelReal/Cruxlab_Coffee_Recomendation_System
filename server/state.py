import sys
import os

# Ensure the root directory is in the path for cross-module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_science.preprocess import load_all_data, preprocess_coffee_data

# Global data holders for the application lifecycle
recipes, users, train, val, test = None, None, None, None, None

async def init_data():
    """
    Initializes the datasets for the application state.
    Captures all 5 datasets from the loader to enable validation lookup.
    """
    global recipes, users, train, val, test
    # Load raw data and unpack all splits
    r_raw, u_raw, t_raw, v_raw, ts_raw = load_all_data()
    
    # Run standard preprocessing on main sets
    recipes, users, train = preprocess_coffee_data(r_raw, u_raw, t_raw)
    
    # Store validation and test sets for metric calculations
    val = v_raw
    test = ts_raw