import streamlit as st

def inject_custom_css():
    """Injects custom CSS for scannability and larger fonts."""
    st.markdown(
        """
        <style>
            html, body, [class*="css"] { font-size: 20px; }
            .stMarkdown p, .stText, .stSidebar p, .stInfo { font-size: 24px !important; line-height: 1.6; }
            h1 { font-size: 52px !important; }
            h2 { font-size: 44px !important; }
            h3 { font-size: 36px !important; }
            h4 { font-size: 30px !important; }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_sidebar_profile(profile: dict):
    """Renders user sidebar profile."""
    st.sidebar.markdown("---")
    st.sidebar.header("👤 Profile Info")
    
    eq_list = ", ".join(profile['owned_equipment'])
    st.sidebar.info(f"**Equipment:** {eq_list}")

    st.sidebar.markdown("### Taste Preferences")
    for taste, value in profile['preferences'].items():
        st.sidebar.write(f"**{taste.capitalize()}:** {round(value, 2)}")
        st.sidebar.progress(float(value))

def render_recipe_card(recipe: dict, index: int):
    """Renders a single recipe card."""
    column_main, column_btn = st.columns([4, 1])
    column_main.markdown(f"#### ✨ {recipe['name']}")
    
    if column_btn.button("📋 Details", key=f"btn_{index}"):
        st.session_state[f"show_{index}"] = not st.session_state.get(f"show_{index}", False)

    if st.session_state.get(f"show_{index}", False):
        with st.container(border=True):
            st.info(f"💡 **Why?** {recipe['details']['justification']}")
            
            detail_col1, detail_col2 = st.columns(2)
            with detail_col1:
                st.write("**Flavor Profile:**")
                for flavor, val in recipe['details']['flavor'].items():
                    if flavor != 'strength':
                        st.write(f"{flavor.capitalize()}")
                        st.progress(float(val))
            
            with detail_col2:
                st.write("**Technical Specs:**")
                st.write(f"🎯 **Recommendation Score:** `{recipe['score']}`")
                st.write(f"**Required Equipment:** {', '.join(recipe['details']['equipment'])}")
                st.write(f"**Estimated Strength:** {recipe['details']['flavor']['strength']}")
        st.markdown("---")