import streamlit as st
import openai
from streamlit_option_menu import option_menu

# Page config
st.set_page_config(
    page_title="Week 3: AI Personas",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    body {
        color: #5c4033;
        background-color: #f5e6d3;
    }
    .stApp {
        background-image: url('https://images.pexels.com/photos/1484776/pexels-photo-1484776.jpeg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    .stButton>button {
        color: #f5e6d3;
        background-color: #8b4513;
        border: 2px solid #5c4033;
    }
    .stTextInput>div>div>input {
        color: #5c4033;
        background-color: #f5e6d3;
    }
    .stTextArea>div>div>textarea {
        color: #5c4033;
        background-color: #f5e6d3;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None

# Main content
st.title("📚 Week 3: AI Personas")

# Sidebar for API key
with st.sidebar:
    st.header("🤗 OpenAI Authentication")
    openai_api_key = st.text_input(
        "Enter your OpenAI API key:",
        type="password",
        help="Get your API key from https://platform.openai.com/account/api-keys"
    )
    if openai_api_key:
        openai.api_key = openai_api_key
        if not openai_api_key.startswith('sk-'):
            st.warning('Please enter a valid OpenAI API key!', icon='⚠️')
        else:
            st.success('API key successfully loaded!', icon='✅')

# Create tabs for different personas
tab1, tab2 = st.tabs(["AI Archaeologist", "Indiana Jones"])

with tab1:
    st.header("Cultural Heritage Preservation and Oral History Curator Agent")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload an image of cultural artifact", 
            type=['png', 'jpg', 'jpeg'],
            key="archaeologist_uploader"
        )
        if uploaded_file is not None:
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    with col2:
        text_input = st.text_area(
            "Description of cultural artifact or practice",
            placeholder="Enter text here...",
            key="archaeologist_input"
        )
    
    if st.button("Process Cultural Material", key="archaeologist_button"):
        if not openai_api_key:
            st.error("Please enter your OpenAI API key in the sidebar first!")
        elif uploaded_file is None and not text_input:
            st.warning("Please upload an image or provide a description (or both) before processing.")
        else:
            with st.spinner("Analyzing cultural material..."):
                # Your existing archaeologist system prompt and processing logic here
                System_Prompt = """[Your existing archaeologist prompt]"""
                user_message = ""
                if uploaded_file is not None:
                    user_message += "An image of a cultural artifact has been uploaded. "
                if text_input:
                    user_message += f"Description: {text_input}"

                try:
                    chat = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {'role': 'system', 'content': System_Prompt},
                            {'role': 'user', 'content': user_message}
                        ]
                    )
                    response = chat.choices[0].message.content
                    st.success("Analysis completed successfully!")
                    st.subheader("Cultural Analysis:")
                    st.write(response)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

with tab2:
    st.header("Indiana Jones: The Relic Hunter")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Present Your Artifact for Examination", 
            type=['png', 'jpg', 'jpeg'],
            key="indy_uploader"
        )
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Artifact in Review", use_column_width=True)
    
    with col2:
        text_input = st.text_area(
            "Describe the Mystery of this Cultural Treasure",
            placeholder="Share what you know here...",
            key="indy_input"
        )
    
    if st.button("Unlock the Secrets", key="indy_button"):
        if not openai_api_key:
            st.error("Please enter your OpenAI API key in the sidebar first!")
        elif uploaded_file is None and not text_input:
            st.warning("A true adventurer brings evidence—upload an image or share a description!")
        else:
            with st.spinner("Analyzing artifact..."):
                # Your existing Indiana Jones system prompt and processing logic here
                System_Prompt = """[Your existing Indiana Jones prompt]"""
                user_message = ""
                if uploaded_file is not None:
                    user_message += "Artifact Image Submitted for Analysis. "
                if text_input:
                    user_message += f"Artifact Background: {text_input}"

                try:
                    chat = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {'role': 'system', 'content': System_Prompt},
                            {'role': 'user', 'content': user_message}
                        ]
                    )
                    response = chat.choices[0].message.content
                    st.success("Discovery Unveiled!")
                    st.subheader("The Artifact's Secrets Revealed:")
                    st.write(response)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
