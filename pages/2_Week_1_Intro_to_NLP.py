import streamlit as st
from pathlib import Path
import nbformat
from nbconvert import HTMLExporter
import base64
import os

# Set page config
st.set_page_config(
    page_title="Week 1: Intro to NLP",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_notebook_from_path(notebook_path):
    """Load a notebook file from path"""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def render_colab_badge(notebook_path):
    """Generate Google Colab badge with link"""
    colab_link = f"https://colab.research.google.com/github/your-username/AI_Bootcamp_Portfolio/blob/main/assets/{os.path.basename(notebook_path)}"
    badge_url = "https://colab.research.google.com/assets/colab-badge.svg"
    
    return f'[![Open In Colab]({badge_url})]({colab_link})'

def main():
    st.title("Week 1: Introduction to NLP")
    st.markdown("""
    This page contains the interactive notebook for Week 1's introduction to Natural Language Processing.
    You can view the notebook below or open it in Google Colab to run it interactively.
    """)
    
    # Path to the notebook in assets folder
    notebook_path = Path("assets/0_NLP_TextPreprocessing.ipynb")
    
    if notebook_path.exists():
        # Display Colab badge
        st.markdown(render_colab_badge(notebook_path), unsafe_allow_html=True)
        
        try:
            # Load and display notebook
            notebook = load_notebook_from_path(notebook_path)
            html_exporter = HTMLExporter()
            html_exporter.template_name = 'classic'
            
            # Add custom CSS for better notebook display
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
            
            body, _ = html_exporter.from_notebook_node(notebook)
            html_content = custom_css + body
            
            # Display notebook content
            st.components.v1.html(html_content, height=800, scrolling=True)
            
        except Exception as e:
            st.error(f"Error loading notebook: {str(e)}")
    else:
        st.error("Notebook file not found in assets folder")

if __name__ == "__main__":
    main()
