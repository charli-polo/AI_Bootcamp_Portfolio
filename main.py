# main.py
import streamlit as st
import time
import pandas as pd
import numpy as np

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

def caching_demo():
    """Demonstrate Streamlit caching capabilities"""
    st.subheader("🚀 Caching in Streamlit")
    
    # Introduction to Caching
    st.markdown("""
    ### What is Caching?
    Caching is a technique to store the results of expensive computations so they can be reused 
    without having to recompute them. In Streamlit, caching is particularly important because:
    
    1. Streamlit reruns your entire script on any widget interaction
    2. Some operations (like loading data or ML models) are expensive
    3. Multiple users might need the same data/computations
    
    ### How Caching Works in Streamlit
    Streamlit offers two main types of caching:
    
    1. **`st.cache_data`**: For caching data computations
        - Creates a new copy of the data each time
        - Safe for DataFrames, lists, and other data objects
        - Each user gets their own copy of the data
    
    2. **`st.cache_resource`**: For caching global resources
        - Shares the exact same object across all users
        - Perfect for ML models and database connections
        - No data copying, just reference sharing
    
    When a cached function is called, Streamlit:
    1. Checks if the function was called before with the same inputs
    2. If yes, returns the cached result
    3. If no, runs the function and stores the result in cache
    """)
    
    # Example 1: Basic Data Caching
    st.markdown("""
    ### 1. Basic Data Caching Demo
    This example shows how `st.cache_data` works with a DataFrame:
    - First click: Takes 2 seconds (simulated data loading)
    - Subsequent clicks: Instant response (using cached data)
    - Each user gets a fresh copy of the DataFrame
    """)
    
    @st.cache_data
    def load_data():
        time.sleep(2)  # Simulate slow data loading
        return pd.DataFrame(
            np.random.randn(1000, 5),
            columns=['A', 'B', 'C', 'D', 'E']
        )
    
    if st.button("Load Data"):
        with st.spinner("Loading data..."):
            df = load_data()
            st.success("Data loaded! Notice how it's instant on subsequent loads.")
            st.dataframe(df.head())
    
    # Example 2: Parameterized Caching
    st.markdown("""
    ### 2. Parameterized Caching Demo
    This example demonstrates how caching works with function parameters:
    - Cache is unique for each input value
    - Changing the slider creates a new cache entry
    - Same slider value reuses cached result
    
    Try moving the slider and clicking compute multiple times to see how 
    Streamlit caches different results for different inputs!
    """)
    
    @st.cache_data
    def expensive_computation(n):
        time.sleep(2)  # Simulate complex computation
        return [i * 2 for i in range(n)]
    
    n = st.slider("Select number of items", 1, 10, 5)
    if st.button("Compute"):
        with st.spinner("Computing..."):
            result = expensive_computation(n)
            st.success("Computation complete! Cached for this input value.")
            st.write(result)
    
    # Example 3: Resource Caching
    st.markdown("""
    ### 3. Resource Caching Demo (ML Models)
    This example shows how `st.cache_resource` works:
    - Perfect for ML models, database connections
    - Single instance shared across all users
    - Stays in memory until the app restarts
    
    In a real app, this would load a machine learning model:
    ```python
    @st.cache_resource
    def load_model():
        model = tf.keras.models.load_model('my_model.h5')
        return model
    ```
    """)
    
    @st.cache_resource
    def load_ml_model():
        time.sleep(3)  # Simulate model loading
        return {"name": "Demo Model", "version": "1.0"}
    
    if st.button("Load ML Model"):
        with st.spinner("Loading model..."):
            model = load_ml_model()
            st.success("Model loaded! This is cached across all sessions.")
            st.json(model)
    
    # Cache Management
    st.markdown("""
    ### ⚙️ Cache Management
    Streamlit provides ways to clear different types of caches:
    - Clear data cache when you want fresh data
    - Clear resource cache to reload models/connections
    - Each cached function can also clear its own cache
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Data Cache"):
            st.cache_data.clear()
            st.success("Data cache cleared!")
    
    with col2:
        if st.button("Clear Resource Cache"):
            st.cache_resource.clear()
            st.success("Resource cache cleared!")
    
    # Caching Best Practices
    with st.expander("📚 Caching Best Practices"):
        st.markdown("""
        ### When to Use st.cache_data
        - DataFrame operations and transformations
        - API calls and web scraping results
        - Data preprocessing steps
        - Feature engineering computations
        - Any function that returns data objects
        
        ### When to Use st.cache_resource
        - ML model loading and initialization
        - Database connections
        - Global resources (like API clients)
        - Heavy computation results shared across users
        
        ### Tips for Effective Caching
        1. **Cache Expensive Operations**
           - Only cache operations that take significant time
           - Don't cache simple calculations
        
        2. **Choose the Right Cache Type**
           - Use `st.cache_data` for data operations
           - Use `st.cache_resource` for shared resources
        
        3. **Handle Cache Invalidation**
           - Clear cache when data needs refreshing
           - Use TTL for time-sensitive data
           - Consider memory usage for large datasets
        
        4. **Avoid Common Pitfalls**
           - Don't cache functions with side effects
           - Be careful with mutable objects in `st.cache_resource`
           - Watch out for memory usage with large caches
        
        ### Advanced Caching Features
        - **TTL (Time To Live)**: `@st.cache_data(ttl="1h")`
        - **Max Entries**: `@st.cache_data(max_entries=100)`
        - **Show Spinner**: `@st.cache_data(show_spinner="Loading...")`
        
        [📚 Read more about caching](https://docs.streamlit.io/library/advanced-features/caching)
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
    
    tab1, tab2, tab3, tab4 = st.tabs(["Tutorial", "Features", "Best Practices", "Caching Demo"])
    
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
    
    with tab4:
        caching_demo()
    
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