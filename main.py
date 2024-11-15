# main.py
import streamlit as st
import nbformat
from nbconvert import HTMLExporter
import io
import os
from pathlib import Path
import base64

# Set page config
st.set_page_config(
    page_title="AI Toolkit",
    layout="wide",  # Changed from page_layout to layout
    initial_sidebar_state="expanded",
)

# Initialize session state
if 'recent_files' not in st.session_state:
    st.session_state.recent_files = []

def load_notebook(file):
    """Load a notebook file and return nbformat notebook object"""
    content = file.read()
    if isinstance(content, bytes):
        content = content.decode('utf-8')
    return nbformat.reads(content, as_version=4)

def convert_notebook_to_html(notebook):
    """Convert notebook to HTML using nbconvert"""
    html_exporter = HTMLExporter()
    html_exporter.template_name = 'classic'
    body, _ = html_exporter.from_notebook_node(notebook)
    
    custom_css = """
    <style>
        .notebook-viewer {
            padding: 20px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .input_area {
            background-color: #f8f9fa;
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
        }
        .output_area {
            padding: 10px;
            margin: 5px 0;
        }
    </style>
    """
    return custom_css + body

def get_notebook_metadata(notebook):
    """Extract metadata from notebook"""
    metadata = {
        'cells': len(notebook.cells),
        'code_cells': len([c for c in notebook.cells if c.cell_type == 'code']),
        'markdown_cells': len([c for c in notebook.cells if c.cell_type == 'markdown']),
    }
    return metadata

def sidebar_info():
    """Display sidebar information"""
    st.sidebar.title("🚀 AI Toolkit")
    st.sidebar.markdown("""
    Welcome to AI Toolkit! This application provides:
    
    - 📓 Notebook Viewer
    - 📝 About Me
    - 🔍 RAG Implementation
    - 🎯 Model Fine-tuning
    - 🤖 AI Persona
    """)

def main():
    sidebar_info()
    
    st.title("📓 Notebook Viewer")
    st.markdown("""
    Welcome to the AI Toolkit! This is the main page featuring our Notebook Viewer.
    Use the sidebar to navigate to other tools and features.
    """)
    
    # File uploader section
    st.header("Upload Notebook")
    uploaded_file = st.file_uploader("Upload a notebook (.ipynb)", type=['ipynb'])
    
    if uploaded_file:
        try:
            # Load and convert notebook
            notebook = load_notebook(uploaded_file)
            html_content = convert_notebook_to_html(notebook)
            
            # Get metadata
            metadata = get_notebook_metadata(notebook)
            
            # Display metadata in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Cells", metadata['cells'])
            with col2:
                st.metric("Code Cells", metadata['code_cells'])
            with col3:
                st.metric("Markdown Cells", metadata['markdown_cells'])
            
            # Store in recent files
            file_info = {
                'name': uploaded_file.name,
                'metadata': metadata
            }
            if file_info not in st.session_state.recent_files:
                st.session_state.recent_files.insert(0, file_info)
                st.session_state.recent_files = st.session_state.recent_files[:5]
            
            # Display notebook content
            st.components.v1.html(html_content, height=800, scrolling=True)
            
        except Exception as e:
            st.error(f"Error loading notebook: {str(e)}")
    
    # Recent files in sidebar
    if st.session_state.recent_files:
        st.sidebar.markdown("### Recent Notebooks")
        for file_info in st.session_state.recent_files:
            with st.sidebar.expander(file_info['name']):
                st.write(f"Total Cells: {file_info['metadata']['cells']}")
                st.write(f"Code Cells: {file_info['metadata']['code_cells']}")
                st.write(f"Markdown Cells: {file_info['metadata']['markdown_cells']}")

if __name__ == "__main__":
    main()