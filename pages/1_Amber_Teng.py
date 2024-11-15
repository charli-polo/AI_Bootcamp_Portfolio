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
        st.header("Amber Teng")
        st.subheader("AI/ML Engineer")
        st.write("""
        👋 Hello! I'm an AI/ML Engineer passionate about building intelligent systems
        that solve real-world problems.
        """)
        
        # Contact information
        st.markdown("""
        📧 Email: angelamarieteng@gmail.com
                    
        🔗 LinkedIn: [linkedin.com/in/angelavteng](https://www.linkedin.com/in/angelavteng)  
        
        💻 GitHub: [github.com/angelaaaateng](https://github.com/angelaaaateng)
        """)

    
    # Projects section
    st.header("Featured Projects")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            st.subheader("LogiLynk: Logistics Support Chatbot")
            st.write("""
            Built an intelligent chatbot using LangChain, FAISS, and OpenAI for 
            handling logistics queries and package tracking. Features RAG implementation
            for accurate context-aware responses.
            """)
            
    with col2:
        with st.container():
            st.subheader("AI Personas & Cultural Heritage")
            st.write("""
            Developed an AI system with multiple personas (AI Archaeologist & Indiana Jones) 
            for cultural heritage preservation and artifact analysis. Implements OpenAI's 
            GPT-4 for intelligent artifact examination and cultural insights.
            """)

    # Second row of projects
    col3, col4 = st.columns(2)
    with col3:
        with st.container():
            st.subheader("RAG Document Analysis")
            st.write("""
            Created a document analysis system using LangChain's FAISS integration
            for efficient text processing and retrieval. Supports PDF and text documents
            with intelligent chunking and semantic search.
            """)
            
    with col4:
        with st.container():
            st.subheader("AI Persona Creator")
            st.write("""
            Engineered a flexible AI persona creation system with customizable 
            personalities, backgrounds, and knowledge domains. Features persistent
            state management and dynamic prompt generation.
            """)

if __name__ == "__main__":
    main()