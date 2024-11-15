import streamlit as st
from pathlib import Path
import json

st.set_page_config(
    page_title="Week 2: Finetuning",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_notebook(file_path):
    """Load and parse a Jupyter notebook"""
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    return notebook

def display_notebook_cell(cell):
    """Display a single notebook cell"""
    if cell['cell_type'] == 'markdown':
        st.markdown(''.join(cell['source']))
    elif cell['cell_type'] == 'code':
        with st.expander("Show code", expanded=True):
            st.code(''.join(cell['source']), language='python')
        if 'outputs' in cell and cell['outputs']:
            with st.expander("Show output", expanded=False):
                for output in cell['outputs']:
                    if 'text' in output:
                        st.text(''.join(output['text']))
                    elif 'data' in output:
                        if 'text/plain' in output['data']:
                            st.text(output['data']['text/plain'])
                        if 'image/png' in output['data']:
                            st.image(output['data']['image/png'])

def main():
    st.title("📓 Week 2: Finetuning Notebooks")
    
    tab1, tab2 = st.tabs(["Finetuning", "Inference"])
    
    # Load notebooks
    finetuning_notebook = load_notebook("assets/3_Finetuninng_Gemma_2b_it_AT.ipynb")
    inference_notebook = load_notebook("assets/4_Inferencing_Finetuned_Model_AT.ipynb")
    
    with tab1:
        st.header("Finetuning Gemma 2B")
        for cell in finetuning_notebook['cells']:
            display_notebook_cell(cell)
    
    with tab2:
        st.header("Inference with Finetuned Model")
        for cell in inference_notebook['cells']:
            display_notebook_cell(cell)

if __name__ == "__main__":
    main()
