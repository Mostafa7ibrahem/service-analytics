import streamlit as st


def init():
    st.session_state.setdefault("dataframes", {})
    st.session_state.setdefault("kpis", {})
    st.session_state.setdefault("insights", {})
    st.session_state.setdefault("ml_results", {})
    st.session_state.setdefault("correlations", {})
    st.session_state.setdefault("loaded", False)
    st.session_state.setdefault("theme", "light")
    st.session_state.setdefault("filters", {})
