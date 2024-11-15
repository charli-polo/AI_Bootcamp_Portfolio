import streamlit as st
from pathlib import Path
import json
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from huggingface_hub import login
import subprocess
import sys
import pkg_resources
from utils.token_manager import get_huggingface_token, set_huggingface_token, validate_huggingface_token

st.set_page_config(
    page_title="Week 2: Finetuning",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    if "model" not in st.session_state:
        st.session_state.model = None
    if "tokenizer" not in st.session_state:
        st.session_state.tokenizer = None
    if "hf_token" not in st.session_state:
        st.session_state.hf_token = None

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

def get_completion(query: str, model, tokenizer) -> str:
    """Generate completion using the model"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    prompt_template = """
    <start_of_turn>user
    Below is an instruction that describes a task. Write a response that appropriately completes the request.
    {query}
    <end_of_turn>
    <start_of_turn>model

    """
    prompt = prompt_template.format(query=query)

    encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    
    model_inputs = encodeds.to(device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2
    )
    
    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Extract only the model's response
    response = decoded.split("<start_of_turn>model")[-1].strip()
    return response

def load_model():
    """Load the model and tokenizer"""
    try:
        if not st.session_state.hf_token:
            st.error("Please enter your HuggingFace token first!")
            return None, None
            
        # Login to HuggingFace
        login(st.session_state.hf_token)
        
        # First load base model tokenizer
        base_model = "google/gemma-2b-it"
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                token=st.session_state.hf_token,
                trust_remote_code=True
            )
            st.success("✅ Tokenizer loaded successfully")
        except Exception as e:
            st.error(f"Failed to load tokenizer: {str(e)}")
            return None, None
            
        # Then load your finetuned model
        try:
            model_id = "at2507/gemma-2b-instruct-ft-python-code-instructions"
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                token=st.session_state.hf_token,
                trust_remote_code=True
            )
            st.success("✅ Model loaded successfully")
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            return None, None
            
        return model, tokenizer
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please make sure you have installed all required packages")
        return None, None

def main():
    st.title("📓 Week 2: Finetuning Notebooks")
    initialize_session_state()
    
    # Add HuggingFace token input in sidebar
    with st.sidebar:
        st.header("🤗 HuggingFace Authentication")
        hf_token = st.text_input(
            "Enter your HuggingFace token:",
            type="password",
            help="Get your token from https://huggingface.co/settings/tokens"
        )
        if hf_token:
            if validate_huggingface_token(hf_token):
                set_huggingface_token(hf_token)
                st.success("HuggingFace token set successfully!")
            else:
                st.error("Invalid HuggingFace token format")
            
        st.markdown("""
        ### How to get your token:
        1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
        2. Click on "New token"
        3. Give it a name and select "read" access
        4. Copy and paste the token here
        """)
    
    # Changed to only two tabs
    tab1, tab2, tab3 = st.tabs(["Finetuning", "Inference", "Model Card"])
    
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
            
    with tab3:
        st.header("🤖 Gemma 2B Instruct Fine-tuned Model Card")
        
        # Model Overview
        st.markdown("""
        ## Model Overview
        This is a fine-tuned version of Google's Gemma 2B model, specifically optimized for Python code instruction generation.
        
        ### Key Specifications
        - **Base Model:** google/gemma-2b-it
        - **Model Size:** 2.51B parameters
        - **Format:** Safetensors
        - **Tensor Type:** FP16
        """)
        
        # Evaluation Results
        with st.expander("📊 Evaluation Results", expanded=True):
            st.markdown("""
            ### Performance Metrics
            
            #### Perplexity Scores
            - **Base Model:** 53.13
            - **Fine-tuned Model:** 1.99
            
            *A significant reduction in perplexity indicates the model's improved ability to predict Python code tokens after fine-tuning.*
            
            #### BLEU Scores
            - **Base Model:** 0.4488
            - **Fine-tuned Model:** 0.4101
            
            *While BLEU scores show a slight decrease, perplexity improvements suggest better code generation capabilities.*
            
            ### Understanding the Metrics
            
            **Perplexity:**
            - Lower values indicate better prediction capability
            - The dramatic reduction from 53.13 to 1.99 shows significant improvement in the model's understanding of Python code patterns
            - Calculated using sliding window approach for more accurate assessment
            
            **BLEU Score Context:**
            - Measures translation quality between 0 and 1
            - While slightly lower in the fine-tuned model, BLEU scores should be considered alongside other metrics
            - Code generation often has multiple valid solutions, making BLEU scores less definitive for code tasks
            """)
        
        # Model Usage
        with st.expander("🛠️ Usage & Capabilities", expanded=True):
            st.markdown("""
            ### Primary Use Cases
            - Python code generation
            - Code instruction following
            - Technical documentation generation
            
            ### Generation Parameters
            ```python
            generation_config = {
                "max_length": 2500,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50
            }
            ```
            """)
        
        # Technical Details
        with st.expander("⚙️ Technical Details", expanded=True):
            st.markdown("""
            ### Training Configuration
            - Fine-tuned from google/gemma-2b-it base model
            - Optimized for instruction following
            - Uses FP16 precision for efficient inference
            - Evaluation performed on dedicated test set
            
            ### Evaluation Process
            - Models evaluated using multiple metrics (BLEU, ROUGE, METEOR)
            - Test data includes diverse Python coding tasks
            - Evaluation performed with torch.no_grad() for efficiency
            """)
        
        # Links and References
        st.markdown("""
        ### 🔗 Important Links
        - [Model on Hugging Face](https://huggingface.co/at2507/gemma-2b-instruct-ft-python-code-instructions)
        - [Base Model (Gemma 2B)](https://huggingface.co/google/gemma-2b-it)
        """)
        
        # Limitations and Disclaimers
        st.warning("""
        **Limitations & Considerations:**
        - While perplexity shows dramatic improvement, BLEU scores indicate a trade-off in exact matching
        - Model may generate alternative but valid code solutions
        - Performance may vary based on input complexity
        - Best results when used with clear, well-structured prompts
        """)

if __name__ == "__main__":
    main()
