# main.py
import streamlit as st

# Set page config
st.set_page_config(
    page_title="AI Bootcamp Portfolio",
    layout="wide",
    initial_sidebar_state="expanded",
)

def sidebar_info():
    """Display sidebar information"""
    st.sidebar.title("🚀 AI Bootcamp Portfolio")
    st.sidebar.markdown("""
    Welcome to my AI Bootcamp Portfolio! Navigate through:
    
    - 📝 About Me
    - 📚 Week 1: Intro to NLP
    - 🤖 Week 2: Model Fine-tuning
    - 🎯 Week 3: AI Personas
    - 📦 Week 4: RAG Implementation
    """)

def main():
    sidebar_info()
    
    st.title("Welcome to My AI Bootcamp Portfolio!")
    st.markdown("""
    Welcome to my AI Bootcamp Portfolio! This portfolio showcases my journey and projects 
    through the AI Bootcamp. Use the sidebar to navigate through different weeks and projects.
    
    ### Portfolio Contents:
    - **About Me**: Learn about my background and skills
    - **Week 1**: Introduction to Natural Language Processing
    - **Week 2**: Model Fine-tuning Techniques
    - **Week 3**: Building AI Personas
    - **Week 4**: RAG Implementation for Logistics Support
    """)

if __name__ == "__main__":
    main()