# pages/1_📝_About_Me.py
import streamlit as st
from pathlib import Path
import base64

st.set_page_config(
    page_title="About Me",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_profile_image():
    # Using a real placeholder image service
    st.image("assets/Amber.png")

def main():
    # Header
    st.title("About Me")
    
    # Layout with two columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        load_profile_image()
        
    with col2:
        st.header("John Doe")
        st.subheader("AI/ML Engineer")
        st.write("""
        👋 Hello! I'm an AI/ML Engineer passionate about building intelligent systems
        that solve real-world problems.
        """)
        
        # Contact information
        st.markdown("""
        📧 Email: john.doe@example.com  
        🔗 LinkedIn: [linkedin.com/in/johndoe](https://linkedin.com/in/johndoe)  
        💻 GitHub: [github.com/johndoe](https://github.com/johndoe)
        """)
    
    # Skills section
    st.header("Skills")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Programming Languages**
        - Python
        - JavaScript
        - SQL
        """)
        
    with col2:
        st.markdown("""
        **ML/AI**
        - PyTorch
        - TensorFlow
        - Hugging Face
        """)
        
    with col3:
        st.markdown("""
        **Tools & Platforms**
        - Docker
        - AWS
        - Git
        """)
    
    # Experience section
    st.header("Experience")
    with st.expander("Senior ML Engineer - Tech Corp (2020-Present)"):
        st.write("""
        - Led development of NLP systems for customer service automation
        - Implemented and deployed RAG systems for knowledge base search
        - Managed team of 3 ML engineers
        """)
        
    with st.expander("ML Engineer - AI Startup (2018-2020)"):
        st.write("""
        - Developed computer vision models for retail analytics
        - Implemented MLOps practices and CI/CD pipelines
        - Reduced model training time by 40%
        """)
    
    # Projects section
    st.header("Featured Projects")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            st.subheader("RAG System for Documentation")
            st.write("""
            Built a RAG system using LangChain and OpenAI for 
            intelligent documentation search and question answering.
            """)
            
    with col2:
        with st.container():
            st.subheader("Fine-tuned LLM for Code Generation")
            st.write("""
            Fine-tuned CodeBERT model for Python code generation,
            achieving 85% accuracy on test set.
            """)

if __name__ == "__main__":
    main()