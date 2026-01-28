import streamlit as st
import requests

# API Configuration
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
    st.sidebar.header("👤 Profile Info")
    profile = requests.get(f"{BASE_URL}/user/{uid}").json()
    
    st.sidebar.info(f"**Equipment:** {', '.join(profile['owned_equipment'])}")

    # Displaying all taste preferences including Strength
    for taste, val in profile['preferences'].items():
        st.sidebar.text(f"{taste.capitalize()}: {round(val, 2)}")
        st.sidebar.progress(float(val))

except Exception as e:
    st.error(f"Backend Connection Error: {e}")
    st.stop()

# 2. Main Recommendations logic
if st.button("🚀 Get Recommendations"):
    with st.spinner("Analyzing flavor profiles..."):
        res = requests.get(f"{BASE_URL}/recommend/{uid}").json()
        st.session_state['recs'] = res['recommendations']

# 3. Displaying recommendations with a "Details" button
if 'recs' in st.session_state:
    st.subheader("Top Recipes for you")
    
    for i, rec in enumerate(st.session_state['recs']):
        # Creating a layout with columns for Name and Button
        col1, col2 = st.columns([4, 1])
        
        col1.markdown(f"#### ✨ {rec['name']} (Score: {rec['score']})")
        
        # Details button for each recommendation
        if col2.button("📋 Details", key=f"btn_{i}"):
            st.session_state[f"show_details_{i}"] = not st.session_state.get(f"show_details_{i}", False)

        # Expanding details logic
        if st.session_state.get(f"show_details_{i}", False):
            with st.container(border=True):
                st.info(f"💡 **Why?** {rec['details']['justification']}")
                
                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    st.write("**Flavor Profile:**")
                    for flavor, val in rec['details']['flavor'].items():
                        st.text(f"{flavor.capitalize()}: {val}")
                        st.progress(float(val))
                
                with d_col2:
                    st.write("**Required Equipment:**")
                    st.write(", ".join(rec['details']['equipment']))
                    st.write(f"**Target Strength:** {rec['details']['flavor']['strength']}")
            st.markdown("---")