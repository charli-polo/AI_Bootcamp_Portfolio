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

def interactive_demo():
    """Show interactive Streamlit features"""
    with st.expander("🎮 Try Streamlit Features"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Widgets")
            name = st.text_input("Enter your name")
            age = st.slider("Select your age", 0, 100, 25)
            is_happy = st.checkbox("I'm happy!")
            
        with col2:
            st.subheader("Output Display")
            if name:
                st.write(f"Hello, {name}!")
            st.write(f"You are {age} years old")
            if is_happy:
                st.balloons()

def analytics_demo():
    """Demo of Streamlit's built-in analytics capabilities"""
    st.subheader("📊 App Analytics")
    
    with st.expander("How to View App Analytics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### From Your Workspace
            1. Go to share.streamlit.io
            2. Click the ⋮ menu next to your app
            3. Select "Analytics"
            """)
            
        with col2:
            st.markdown("""
            #### From Your App
            1. Click "Manage app" in lower-right corner
            2. Click the ⋮ menu
            3. Select "Analytics"
            """)
    
    # Analytics Features Demo
    st.markdown("""
    ### Available Analytics
    
    #### 👥 Viewer Metrics
    - Total viewer count (from April 2022)
    - Recent unique viewers (last 20)
    - Timestamp of each viewer's last visit
    
    #### 🔐 Privacy Features
    - Public apps: Anonymous viewers shown as pseudonyms
    - Private apps: Authorized viewers shown by email
    - Workspace members can see fellow member identities
    
    #### 📊 Access Control
    - Workspace members see analytics for all public apps
    - Invited viewers gain analytics access
    - Private apps show only authorized viewers
    """)
    
    st.info("""
    **Note:** When you invite a viewer to your app, they can:
    - View app analytics
    - See analytics for all public apps in your workspace
    - Invite additional viewers
    - See emails of developers and other viewers in the workspace
    """)

def main():
    sidebar_info()
    
    st.title("Welcome to My AI Bootcamp Portfolio!")
    
    # Introduction
    st.markdown("""
    This portfolio showcases my journey through the AI Bootcamp and serves as a practical 
    demonstration of Streamlit's capabilities.
    """)
    
    # Streamlit Tutorial
    st.header("🎈 Getting Started with Streamlit")
    
    tab1, tab2, tab3 = st.tabs(["Tutorial", "Features", "Best Practices"])
    
    with tab1:
        st.markdown("""
        ### Quick Streamlit Tutorial
        
        1. **Installation**
        ```bash
        pip install streamlit
        ```
        
        2. **Create Your First App**
        ```python
        import streamlit as st
        st.write("Hello, World!")
        ```
        
        3. **Run Your App**
        ```bash
        streamlit run app.py
        ```
        """)
        
        interactive_demo()
        analytics_demo()
    
    with tab2:
        st.markdown("""
        ### 🛠️ Features Used in This App
        
        #### Layout Elements
        - `st.sidebar`: Navigation menu
        - `st.columns`: Multi-column layouts
        - `st.expander`: Collapsible sections
        - `st.tabs`: Organized content tabs
        
        #### Input Widgets
        - `st.text_input`: Text entry
        - `st.slider`: Numeric input
        - `st.checkbox`: Boolean input
        - `st.selectbox`: Dropdown selection
        
        #### Display Elements
        - `st.markdown`: Formatted text
        - `st.code`: Code blocks
        - `st.metric`: KPI metrics
        - `st.balloons`: Animations
        
        [📚 Complete Streamlit API Reference](https://docs.streamlit.io/library/api-reference)
        """)
    
    with tab3:
        st.markdown("""
        ### 💡 Streamlit Best Practices
        
        #### Code Organization
        - Use functions to organize code
        - Cache expensive computations with `@st.cache_data`
        - Keep the main script clean and modular
        
        #### UI/UX Design
        - Use clear headers and sections
        - Implement responsive layouts
        - Add helpful documentation
        - Include interactive examples
        
        #### Deployment Tips
        1. **Streamlit Cloud Deployment**
           - Connect your GitHub repository
           - Requirements.txt must be up to date
           - Use secrets management for API keys
           - Set appropriate Python version
        
        2. **Performance Optimization**
           - Cache data loading operations
           - Optimize large data operations
           - Use session state for app state management
        
        3. **Security Considerations**
           - Never commit API keys
           - Use st.secrets for sensitive data
           - Implement proper input validation
        
        [🚀 Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/get-started)
        """)
    
    # Additional Resources
    st.header("📚 Additional Resources")
    st.markdown("""
    - [Streamlit Documentation](https://docs.streamlit.io)
    - [Streamlit Gallery](https://streamlit.io/gallery)
    - [Streamlit Components](https://streamlit.io/components)
    - [Streamlit Forum](https://discuss.streamlit.io)
    """)

if __name__ == "__main__":
    main()