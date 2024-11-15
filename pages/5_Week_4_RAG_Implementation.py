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
from langchain.embeddings.base import Embeddings
from typing import List

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

def load_precomputed_embeddings(bot_selection):
    """Load precomputed embeddings from assets folder based on bot selection"""
    try:
        # Select appropriate embeddings file
        if bot_selection == "LogiLynk":
            embeddings_path = "assets/parcel_embeddings.npy"
        else:  # Wells FurGo
            embeddings_path = "assets/cat_logistics_embeddings.npy"
            
        if os.path.exists(embeddings_path):
            embeddings = np.load(embeddings_path)
            st.session_state[f"{bot_selection}_embeddings"] = embeddings
            return True
        else:
            st.error(f"Embeddings file not found at {embeddings_path}")
            return False
    except Exception as e:
        st.error(f"Error loading embeddings: {str(e)}")
        return False

# Add this class after the imports
class PrecomputedEmbeddings(Embeddings):
    """Custom embeddings class for pre-computed embeddings"""
    def __init__(self, embeddings_array):
        self.embeddings = embeddings_array
        self._current_index = 0
        self.embed_dim = len(embeddings_array[0])
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return pre-computed embeddings for documents"""
        batch_size = len(texts)
        embeddings = self.embeddings[self._current_index:self._current_index + batch_size]
        self._current_index += batch_size
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Return pre-computed embedding for query"""
        # For queries, use OpenAI API
        client = get_openai_client()
        text = str(text).replace("\n", " ")
        return client.Embedding.create(
            input=[text],
            model="text-embedding-3-small"
        )['data'][0]['embedding']

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts"""
        return self.embed_documents(texts)

def load_and_process_data(api_key, bot_selection):
    """Load CSV data and create FAISS index"""
    try:
        # Load appropriate CSV data and embeddings based on bot selection
        if bot_selection == "LogiLynk":
            file_path = "assets/Extended_Parcel_Information_Dataset.csv"
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        else:  # Wells FurGo
            file_path = "assets/large_ph_cat_logistics_clean_dataset.csv"
            # Load pre-computed embeddings
            embeddings_path = "assets/cat_logistics_embeddings.npy"
            if os.path.exists(embeddings_path):
                embeddings_array = np.load(embeddings_path)
                embeddings = PrecomputedEmbeddings(embeddings_array)
            else:
                st.error(f"Embeddings file not found at {embeddings_path}")
                return False
        
        # Load CSV data
        loader = CSVLoader(
            file_path=file_path,
            encoding="utf-8",
            csv_args={'delimiter': ','}
        )
        documents = loader.load()
        
        # Create FAISS index
        st.session_state[f"{bot_selection}_index"] = FAISS.from_documents(documents, embeddings)
        st.session_state[f"{bot_selection}_documents"] = documents
        
        return True
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return False

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'LogiLynk_embeddings' not in st.session_state:
    st.session_state.LogiLynk_embeddings = None
if 'Wells FurGo_embeddings' not in st.session_state:
    st.session_state['Wells FurGo_embeddings'] = None
if 'LogiLynk_index' not in st.session_state:
    st.session_state.LogiLynk_index = None
if 'Wells FurGo_index' not in st.session_state:
    st.session_state['Wells FurGo_index'] = None
if 'LogiLynk_documents' not in st.session_state:
    st.session_state.LogiLynk_documents = None
if 'Wells FurGo_documents' not in st.session_state:
    st.session_state['Wells FurGo_documents'] = None

# Move page config to the very top, before any other code
st.set_page_config(
    page_title="Week 4: RAG Implementation",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📦"
)

# Update the CSS to fix padding and spacing
st.markdown("""
<style>
    /* Reset padding/margin */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    
    /* Ensure radio buttons are visible */
    .stRadio > label {
        color: #000000 !important;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .stRadio > div {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Light theme with subtle background */
    .stApp {
        background: linear-gradient(
            rgba(255, 255, 255, 0.9),
            rgba(255, 255, 255, 0.9)
        ),
        url('https://images.pexels.com/photos/4483610/pexels-photo-4483610.jpeg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* Make all text black and visible */
    .stApp, [data-testid="stSidebarContent"], .element-container, p, h1, h2, h3, label {
        color: #000000 !important;
    }
    
    /* Add spacing between elements */
    .stRadio {
        margin-top: 2rem;
        margin-bottom: 2rem;
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

# Add a title at the top of the page
st.title("🤖 AI Logistics Assistant")

# Then the bot selection
bot_selection = st.radio(
    "Select Chatbot:",
    ["LogiLynk", "Wells FurGo"],
    horizontal=True,
    key="bot_selection"
)

# Then the sidebar section
with st.sidebar:
    st.header("🔑 OpenAI Authentication")
    api_key = st.text_input("OpenAI API Key:", type="password")
    if api_key:
        if init_openai(api_key):
            st.success("OpenAI API key set successfully!")
            # Initialize data and embeddings with API key and bot selection
            if load_and_process_data(api_key, bot_selection):
                st.success("Data loaded successfully!")
            else:
                st.error("Failed to load data")
        else:
            st.error("Invalid OpenAI API key")

# Rest of the sidebar (Data viewer)
with st.sidebar:
    st.markdown("---")
    st.header("📊 Data Viewer")
    show_data = st.toggle("Show Dataset", value=False)
    
    if show_data and st.session_state[f"{bot_selection}_documents"] is not None:
        try:
            # Convert documents to DataFrame
            data = []
            for doc in st.session_state[f"{bot_selection}_documents"]:
                content_dict = {}
                for line in doc.page_content.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        content_dict[key.strip()] = value.strip()
                data.append(content_dict)
            
            df = pd.DataFrame(data)
            
            # Display DataFrame with styling
            st.dataframe(
                df,
                use_container_width=True,
                height=300,
                hide_index=True
            )
            
            # Add download button with appropriate filename
            csv = df.to_csv(index=False).encode('utf-8')
            filename = "parcel_data.csv" if bot_selection == "LogiLynk" else "cat_logistics_data.csv"
            st.download_button(
                "Download Dataset",
                csv,
                filename,
                "text/csv",
                key=f'download-csv-{bot_selection}'
            )
            
        except Exception as e:
            st.error(f"Error displaying data: {str(e)}")
    elif show_data:
        st.info("Please load data first by entering a valid API key.")

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
Role
You are Wells FurGo, a knowledgeable and compassionate logistics support chatbot specializing in the transport and delivery of cat-related products and live cats in the Philippines.
Your goal is to provide accurate, concise, and empathetic information about shipment tracking, delivery status, shipping costs, and common delivery issues.
Your tone is warm, professional, and supportive, ensuring customers feel informed and reassured during every interaction. Also, use gen-z cat memes in conversation, as appropriate. Use cat-related puns as well.

Instructions
Shipment Tracking: When a customer asks about their shipment, request the tracking number, retrieve the latest status, and provide clear updates on the current location, condition, and estimated delivery date.
Status Explanations: For inquiries about delivery statuses such as "In Transit" or "Delayed," offer simple, friendly explanations. Use relatable terms, and analogies if needed, to help customers understand the process.
Cost Calculations: Guide customers in providing details like product weight, dimensions, and destination for calculating shipping costs. Provide an estimate based on the provided information and explain any factors influencing the cost.
Issue Resolution: For issues such as delays, incorrect addresses, or lost shipments, respond with empathy. Explain next steps clearly, including any proactive measures taken to resolve or escalate the issue.
Proactive Alerts: Offer customers the option to receive notifications about key updates, such as when shipments reach major checkpoints or encounter delays.
FAQ Handling: Address frequently asked questions about handling live cats, special packaging requirements, and preferred delivery times with clarity and simplicity.
Tone and Language: Maintain a professional and caring tone, particularly when discussing delays or challenges. Show understanding and reassurance.

Context
Wells FurGo serves as the primary customer service chatbot for a logistics company specializing in cat-related products and live cat transportation in the Philippines. You handle interactions with pet owners, cat cafés, and retail businesses that often have urgent concerns about the well-being and timely delivery of their cat shipments. Providing accurate and clear updates, coupled with empathy, is crucial for building trust and confidence, especially in sensitive situations like live animal transport.

Constraints
Privacy: Never disclose personal information beyond what has been verified and confirmed by the customer. Always ask for consent before discussing details about shipments.
Conciseness: Ensure responses are clear and concise, avoiding logistics jargon unless necessary for context.
Empathy in Communication: When addressing delays or challenges, prioritize empathy and acknowledge the customer's concern. Provide next steps and reassurance.
Accuracy: Ensure all tracking updates, cost estimates, and shipment details are accurate and up-to-date.
Jargon-Free Language: Use simple language to explain logistics terms or processes to customers, particularly when dealing with pet transport.

Examples
Shipment Tracking Inquiry

Customer: "Where is my cat's shipment?"
Wells FurGo: "I'd be happy to help! Could you please share your tracking number? I'll find the latest update for you, including the current location and estimated delivery date."

Explanation of Delivery Status

Customer: "What does 'Delayed' mean for my cat's shipment?"
Wells FurGo: "A 'Delayed' status means that your cat's shipment has experienced a temporary hold-up, which could be due to factors like weather or logistical constraints. Rest assured, we’re monitoring the situation and will update you as soon as there’s progress."

Cost Calculation Inquiry

Customer: "How much will it cost to ship cat food weighing 5 kg to Cebu?"
Wells FurGo: "I'd be glad to calculate that for you! Could you also share the package dimensions? Once I have those details, I’ll provide an estimate and explain any options for standard or expedited delivery."

Issue Resolution for Delayed Live Cat Shipment

Customer: "I’m worried about my cat's delayed shipment."
Wells FurGo: "I understand your concern, and I’m here to help. Let me check the latest status of your shipment. If needed, we’ll coordinate with the carrier to ensure your cat’s safety and provide you with updates along the way."

Proactive Update Offer

Customer: "Can I get updates on my cat shipment's status?"
Wells FurGo: "Absolutely! I can send you notifications whenever your cat’s shipment reaches a checkpoint or if there are any major updates. Would you like to set that up?"

"""

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
    placeholder = "e.g., What route is my cat product taking to reach Bacolod? the tracking number for my cat product is CAT-LIT-003"

# Chat input
user_input = st.text_input("Enter your message:", placeholder=placeholder)

# Check for API key and process input
if user_input and api_key and st.session_state[f"{bot_selection}_index"] is not None:
    with st.spinner("Processing your request..."):
        try:
            # Use the correct index for the selected bot
            similar_docs = st.session_state[f"{bot_selection}_index"].similarity_search(user_input, k=2)
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