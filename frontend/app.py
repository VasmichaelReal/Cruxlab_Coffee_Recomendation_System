import streamlit as st
import api_client as api
import ui_components as ui

st.set_page_config(page_title="Coffee AI Coach", page_icon="☕", layout="wide")
st.title("☕ Coffee AI Coach")

# 1. User Selection
users = api.fetch_users()
if users:
    user_options = {
        (f"{u['user_id']} (❄️ Cold)" if u['is_cold'] else u['user_id']): u['user_id']
        for u in users
    }
    selected_label = st.sidebar.selectbox("Select User:", options=list(user_options.keys()))
    uid = user_options[selected_label]

    # 2. Render Sidebar Profile
    profile_data = api.fetch_profile(uid)
    if profile_data:
        ui.render_sidebar_profile(profile_data)

    # 3. Recommendation Logic
    if st.button("🚀 Get Recommendations"):
        with st.spinner("Analyzing..."):
            res = api.fetch_recommendations(uid)
            if res:
                st.session_state['recs'] = res['recommendations']

    # 4. Display Results
    if 'recs' in st.session_state:
        st.subheader("Top Recipes for you")
        for i, rec in enumerate(st.session_state['recs']):
            ui.render_recipe_card(rec, i)
else:
    st.warning("No users found. Please check if the Server is running.")