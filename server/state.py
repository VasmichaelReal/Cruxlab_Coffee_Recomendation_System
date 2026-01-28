import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_science.preprocess import load_all_data, preprocess_coffee_data

# Global data holders
recipes, users, train = None, None, None

async def init_data():
    """Initializes the datasets for the application state."""
    global recipes, users, train
    r_raw, u_raw, t_raw, _, _ = load_all_data()
    recipes, users, train = preprocess_coffee_data(r_raw, u_raw, t_raw)