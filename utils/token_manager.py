import streamlit as st
import openai

@st.cache_resource
def get_openai_client():
    """Get cached OpenAI client"""
    if "openai_client" not in st.session_state:
        st.session_state.openai_client = None
    return st.session_state.openai_client

def init_openai(api_key):
    """Initialize OpenAI client with API key and cache it"""
    if not api_key:
        return None
        
    if not api_key.startswith('sk-'):
        return None
        
    try:
        openai.api_key = api_key
        st.session_state.openai_client = openai
        return openai
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        return None

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