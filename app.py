import streamlit as st
import requests

st.set_page_config(page_title="Coffee AI Coach", page_icon="☕")

st.title("☕ Coffee AI Recommender")
st.write("Enter your User ID to get personalized recipes based on your taste and equipment.")

user_id = st.text_input("User ID", placeholder="e.g., user_123")

if st.button("Get My Coffee Recipes"):
    if user_id:
        with st.spinner('Thinking about the perfect brew...'):
            try:
                # Звертаємося до нашого бекенду
                response = requests.get(f"http://localhost:8000/recommend/{user_id}")
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Top {len(data['recommendations'])} recommendations for you:")
                    
                    for rec in data['recommendations']:
                        with st.expander(f"🌟 {rec['name']} (Score: {rec['score']})"):
                            st.write(f"**Recipe ID:** {rec['recipe_id']}")
                            st.write(f"**Required Equipment:** {', '.join(rec['equipment'])}")
                else:
                    st.error("User not found or API error.")
            except Exception as e:
                st.error(f"Connection failed: {e}")
    else:
        st.warning("Please enter a User ID.")