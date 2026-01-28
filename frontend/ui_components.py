import streamlit as st

def render_sidebar_profile(profile):
    """Renders the user profile card and taste progress bars."""
    st.sidebar.markdown("---")
    st.sidebar.header("👤 Profile Info")
    
    # Equipment info
    eq_list = ", ".join(profile['owned_equipment'])
    st.sidebar.info(f"**Equipment:** {eq_list}")

    # Taste preferences with progress bars
    for taste, val in profile['preferences'].items():
        st.sidebar.text(f"{taste.capitalize()}: {round(val, 2)}")
        st.sidebar.progress(float(val))

def render_recipe_card(rec, index):
    """Renders a single recommendation with a 'Details' button."""
    col1, col2 = st.columns([4, 1])
    col1.markdown(f"#### ✨ {rec['name']} (Score: {rec['score']})")
    
    # Toggle state for details
    btn_key = f"btn_{index}"
    if col2.button("📋 Details", key=btn_key):
        st.session_state[f"show_{index}"] = not st.session_state.get(f"show_{index}", False)

    if st.session_state.get(f"show_{index}", False):
        with st.container(border=True):
            st.info(f"💡 **Why?** {rec['details']['justification']}")
            
            d1, d2 = st.columns(2)
            with d1:
                st.write("**Flavor Profile:**")
                for flavor, val in rec['details']['flavor'].items():
                    st.text(f"{flavor.capitalize()}: {val}")
                    st.progress(float(val))
            with d2:
                st.write("**Details:**")
                st.write(f"**Equipment:** {', '.join(rec['details']['equipment'])}")
                st.write(f"**Strength:** {rec['details']['flavor']['strength']}")
        st.markdown("---")