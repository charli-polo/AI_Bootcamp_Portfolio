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

# Initialize OpenAI client with API key
def init_openai(api_key):
    openai.api_key = api_key
    return openai

# Function to get embeddings using OpenAI API v0.28.1
def get_embedding(text, model="text-embedding-3-small"):
    text = str(text).replace("\n", " ")  # Ensure text is string and clean
    return openai.Embedding.create(
        input=[text],
        model=model
    )['data'][0]['embedding']

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

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'documents' not in st.session_state:
    st.session_state.documents = None

# Main content
st.title("📦 LogiLynk: Logistics Support Chatbot")

# Sidebar for API key
with st.sidebar:
    st.header("🔑 OpenAI Authentication")
    openai_api_key = st.text_input(
        "Enter your OpenAI API key:",
        type="password",
        help="Get your API key from https://platform.openai.com/account/api-keys"
    )
    if openai_api_key:
        # Initialize OpenAI with the API key
        openai_client = init_openai(openai_api_key)
        
        if not openai_api_key.startswith('sk-'):
            st.warning('Please enter a valid OpenAI API key!', icon='⚠️')
        else:
            st.success('API key successfully loaded!', icon='✅')
            
            # Load data and create embeddings if not already done
            if 'embeddings' not in st.session_state or st.session_state.embeddings is None:
                try:
                    with st.spinner("Loading parcel data and creating embeddings..."):
                        # Load the dataset
                        dataframed = pd.read_csv('https://raw.githubusercontent.com/ALGOREX-PH/Day-4-AI-First-Dataset-Live/refs/heads/main/Parcel_Information_Dataset.csv')
                        
                        # Create combined text field
                        dataframed['combined'] = dataframed.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
                        texts = dataframed['combined'].tolist()
                        
                        # Initialize embeddings object
                        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                        
                        # Create FAISS index directly from texts
                        search_index = FAISS.from_texts(
                            texts=texts,
                            embedding=embeddings
                        )
                        
                        st.session_state.index = search_index
                        st.success("✅ Data loaded and embeddings created!")
                except Exception as e:
                    st.error(f"Error creating embeddings: {str(e)}")

# Chat interface
st.write("Welcome to LogiLynk! I'm here to help you track packages and answer shipping questions. How can I assist you today?")

# System prompt
System_Prompt = """
Role: You are LogiLynk, a knowledgeable and empathetic logistics support chatbot specializing in assisting customers with their parcel inquiries. Your mission is to provide accurate, concise, and reassuring information on parcel tracking, delivery status, shipping costs, and resolving common delivery issues.

Instructions:
1. Use the provided context to answer customer queries accurately
2. Maintain a professional yet friendly tone
3. If tracking information is available, provide detailed status updates
4. For shipping inquiries, provide clear and accurate information
5. Always prioritize customer satisfaction while being truthful and precise
"""

# Chat input
user_input = st.text_input("Enter your message:", placeholder="e.g., Where is my package? or Track my parcel...")

if user_input and openai_api_key and st.session_state.index is not None:
    with st.spinner("Processing your request..."):
        try:
            # Create query embedding and search
            similar_docs = st.session_state.index.similarity_search(user_input, k=2)
            context = ' '.join([doc.page_content for doc in similar_docs])
            
            # Create structured prompt
            structured_prompt = f"Context:\n{context}\n\nQuery:\n{user_input}\n\nResponse:"
            
            # Get completion from OpenAI using v0.28.1 format
            chat = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": System_Prompt},
                    {"role": "user", "content": structured_prompt}
                ]
            )
            response = chat.choices[0].message['content']
            
            # Display response
            st.write("### LogiLynk Response:")
            st.write(response)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

elif not openai_api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to continue.")