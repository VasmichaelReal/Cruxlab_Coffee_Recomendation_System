import streamlit as st
import requests

BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Coffee AI Coach", page_icon="☕", layout="wide")
st.title("☕ Coffee AI Coach")

# 1. Fetching user list
try:
    users_res = requests.get(f"{BASE_URL}/users").json()
    user_options = {
        (f"{u['user_id']} (❄️ Cold)" if u['is_cold'] else u['user_id']): u['user_id'] 
        for u in users_res['users']
    }
    
    selected_label = st.sidebar.selectbox("Select User:", options=list(user_options.keys()))
    uid = user_options[selected_label]
    
    # User Profile Card in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("📋 Profile Info")
    profile = requests.get(f"{BASE_URL}/user/{uid}").json()
    st.sidebar.info(f"**Equipment:** {', '.join(profile['owned_equipment'])}")
    
    for taste, val in profile['preferences'].items():
        if taste != 'strength':
            st.sidebar.text(f"{taste.capitalize()}: {val}")
            st.sidebar.progress(float(val))

except Exception as e:
    st.error(f"Backend Connection Error: {e}")
    st.stop()

# 2. Main Recommendations logic
if st.button("🚀 Get Recommendations"):
    res = requests.get(f"{BASE_URL}/recommend/{uid}")
    if res.status_code == 200:
        data = res.json()
        st.subheader(f"Top Recipes for you")
        
        for rec in data['recommendations']:
            with st.expander(f"✨ {rec['name']} (Score: {rec['score']})"):
                st.info(f"💡 **Why?** {rec['details']['justification']}")
                st.write(f"**Equipment needed:** {', '.join(rec['details']['equipment'])}")
                st.write("**Flavor Profile:**")
                st.json(rec['details']['flavor'])
    else:
        st.error("Failed to fetch recommendations.")