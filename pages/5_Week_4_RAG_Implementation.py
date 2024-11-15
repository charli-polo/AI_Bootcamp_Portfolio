import streamlit as st
import openai
import numpy as np
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
import warnings
import os
from utils.token_manager import get_openai_client, init_openai

# Initialize OpenAI client with API key
def init_openai(api_key=None):
    """Initialize OpenAI client with API key"""
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

# Function to get embeddings using OpenAI API
def get_embedding(text, model="text-embedding-3-small"):
    client = get_openai_client()
    if not client:
        st.warning("Please enter your OpenAI API token")
        st.stop()
        
    text = str(text).replace("\n", " ")
    return client.Embedding.create(
        input=[text],
        model=model
    )['data'][0]['embedding']

def load_and_process_data(api_key):
    """Load CSV data and create FAISS index"""
    if st.session_state.index is None:
        try:
            # Load CSV data
            loader = CSVLoader(
                file_path="assets/Extended_Parcel_Information_Dataset.csv",
                encoding="utf-8",
                csv_args={'delimiter': ','}
            )
            documents = loader.load()
            
            # Create embeddings with API key
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            
            # Create FAISS index
            st.session_state.index = FAISS.from_documents(documents, embeddings)
            st.session_state.documents = documents
            
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    return True

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'documents' not in st.session_state:
    st.session_state.documents = None

# Page config with light theme
st.set_page_config(
    page_title="Week 4: RAG Implementation",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📦",
    menu_items=None
)

# Custom CSS
st.markdown("""
<style>
    /* Remove top padding/margin */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    
    /* Light theme with subtle background */
    .stApp {
        background: linear-gradient(
            rgba(255, 255, 255, 0.85),
            rgba(255, 255, 255, 0.85)
        ),
        url('https://images.pexels.com/photos/4483610/pexels-photo-4483610.jpeg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* Make all text black */
    .stApp, [data-testid="stSidebarContent"], .element-container, p, h1, h2, h3, label {
        color: #000000 !important;
    }
    
    /* Style for input fields */
    .stTextInput>div>div>input {
        color: #000000;
        background-color: white;
    }

    /* Warning/Info messages */
    .stAlert {
        color: #000000;
        background-color: rgba(255, 255, 255, 0.9);
    }

    /* Hide hamburger menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Sidebar for API key and data loading
with st.sidebar:
    st.header("🔑 OpenAI Authentication")
    api_key = st.text_input("OpenAI API Key:", type="password")
    if api_key:
        if init_openai(api_key):
            st.success("OpenAI API key set successfully!")
            # Initialize data and embeddings with API key
            if load_and_process_data(api_key):
                st.success("Data loaded successfully!")
            else:
                st.error("Failed to load data")
        else:
            st.error("Invalid OpenAI API key")

# Add tab selection at the top
bot_selection = st.radio(
    "Select Chatbot:",
    ["LogiLynk", "Wells FurGo"],
    horizontal=True
)

# System prompts
LOGILYNK_PROMPT = """
Role: You are LogiLynk, a knowledgeable and empathetic logistics support chatbot specializing in assisting customers with their parcel inquiries. Your mission is to provide accurate, concise, and reassuring information on parcel tracking, delivery status, shipping costs, and resolving common delivery issues.

Instructions:
1. Use the provided context to answer customer queries accurately
2. Maintain a professional yet friendly tone
3. If tracking information is available, provide detailed status updates
4. For shipping inquiries, provide clear and accurate information
5. Always prioritize customer satisfaction while being truthful and precise
"""

WELLS_FURGO_PROMPT = """
Role: You are Wells FurGo, a knowledgeable and compassionate logistics support chatbot specializing in the transport and delivery of cat-related products and live cats in the Philippines...
""" # (your full Wells FurGo prompt here)

# Main content based on selection
if bot_selection == "LogiLynk":
    st.title("📦 LogiLynk: Logistics Support Chatbot")
    st.write("Welcome to LogiLynk! I'm here to help you track packages and answer shipping questions. How can I assist you today?")
    System_Prompt = LOGILYNK_PROMPT
    placeholder = "e.g., Hello my name is Michael Brown, Where is the parcel I sent Sara Davis?"
else:
    st.title("😺 Wells FurGo: Cat Logistics Support")
    st.write("Meow there! I'm Wells FurGo, your purr-fessional cat logistics expert. How can I assist you today?")
    System_Prompt = WELLS_FURGO_PROMPT
    placeholder = "e.g., I need to ship my cat food to Cebu, or Track my cat's shipment"

# Chat input
user_input = st.text_input("Enter your message:", placeholder=placeholder)

# Check for API key and process input
if user_input and api_key and st.session_state.index is not None:
    with st.spinner("Processing your request..."):
        try:
            # Create query embedding and search
            similar_docs = st.session_state.index.similarity_search(user_input, k=2)
            context = ' '.join([doc.page_content for doc in similar_docs])
            
            # Create structured prompt
            structured_prompt = f"Context:\n{context}\n\nQuery:\n{user_input}\n\nResponse:"
            
            # Get completion from OpenAI using cached client
            client = get_openai_client()
            chat = client.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": System_Prompt},
                    {"role": "user", "content": structured_prompt}
                ]
            )
            response = chat.choices[0].message['content']
            
            # Display response with appropriate heading
            st.write(f"### {bot_selection} Response:")
            st.write(response)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

elif not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to continue.")