import requests
import streamlit as st

BASE_URL = "http://127.0.0.1:8000"

def fetch_users():
    """Retrieves the list of users from the API."""
    try:
        response = requests.get(f"{BASE_URL}/users")
        response.raise_for_status()
        return response.json().get('users', [])
    except Exception as e:
        st.error(f"Failed to fetch users: {e}")
        return []

def fetch_profile(user_id):
    """Retrieves detailed user profile."""
    try:
        response = requests.get(f"{BASE_URL}/user/{user_id}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch profile: {e}")
        return None

def fetch_recommendations(user_id, strategy="hybrid_ml"):
    """Gets personalized recommendations."""
    try:
        response = requests.get(f"{BASE_URL}/recommend/{user_id}", params={"strategy": strategy})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch recommendations: {e}")
        return None