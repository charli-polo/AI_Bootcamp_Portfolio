import streamlit as st

@st.cache_resource
def get_huggingface_token():
    """Get cached HuggingFace token"""
    if "hf_token" not in st.session_state:
        st.session_state.hf_token = None
    return st.session_state.hf_token

@st.cache_resource
def get_openai_token():
    """Get cached OpenAI token"""
    if "openai_token" not in st.session_state:
        st.session_state.openai_token = None
    return st.session_state.openai_token

def set_huggingface_token(token):
    """Set HuggingFace token and clear cache if changed"""
    if st.session_state.hf_token != token:
        st.session_state.hf_token = token
        get_huggingface_token.clear()

def set_openai_token(token):
    """Set OpenAI token and clear cache if changed"""
    if st.session_state.openai_token != token:
        st.session_state.openai_token = token
        get_openai_token.clear()

def validate_openai_token(token):
    """Validate OpenAI token format"""
    return token.startswith('sk-') and len(token) == 51

def validate_huggingface_token(token):
    """Validate HuggingFace token format"""
    return token.startswith('hf_') and len(token) >= 40 