# pages/4_🤖_AI_Persona.py
import streamlit as st
import openai
from datetime import datetime
import json
import os
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="AI Persona Creator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_persona' not in st.session_state:
    st.session_state.current_persona = None
if 'personas' not in st.session_state:
    st.session_state.personas = {}

class AIPersona:
    def __init__(self, name, personality, background, speaking_style, knowledge_domains):
        self.name = name
        self.personality = personality
        self.background = background
        self.speaking_style = speaking_style
        self.knowledge_domains = knowledge_domains
        
    def get_system_prompt(self):
        return f"""You are {self.name}, an AI assistant with the following characteristics:

Personality: {self.personality}
Background: {self.background}
Speaking Style: {self.speaking_style}
Knowledge Domains: {', '.join(self.knowledge_domains)}

Maintain this persona consistently in all interactions. Respond in a way that reflects
your personality and background while staying helpful and informative."""
    
    def to_dict(self):
        return {
            "name": self.name,
            "personality": self.personality,
            "background": self.background,
            "speaking_style": self.speaking_style,
            "knowledge_domains": self.knowledge_domains
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            data["name"],
            data["personality"],
            data["background"],
            data["speaking_style"],
            data["knowledge_domains"]
        )

def save_personas(personas, file_path="personas.json"):
    """Save personas to a JSON file"""
    personas_dict = {name: persona.to_dict() for name, persona in personas.items()}
    with open(file_path, 'w') as f:
        json.dump(personas_dict, f)

def load_personas(file_path="personas.json"):
    """Load personas from a JSON file"""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            personas_dict = json.load(f)
            return {name: AIPersona.from_dict(data) for name, data in personas_dict.items()}
    return {}

def create_chat_message(role, content):
    """Create a chat message with timestamp"""
    return {
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def get_ai_response(messages, persona):
    """Get response from OpenAI API"""
    try:
        # Prepare messages for API
        api_messages = [{"role": "system", "content": persona.get_system_prompt()}]
        api_messages.extend([{"role": m["role"], "content": m["content"]} for m in messages])
        
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=api_messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error getting AI response: {str(e)}")
        return None

def create_persona_form():
    """Create a new persona using a form"""
    with st.form("create_persona"):
        st.subheader("Create New Persona")
        name = st.text_input("Name")
        personality = st.text_area("Personality Description")
        background = st.text_area("Background Story")
        speaking_style = st.text_area("Speaking Style")
        knowledge_domains = st.text_input("Knowledge Domains (comma-separated)")
        
        submitted = st.form_submit_button("Create Persona")
        
        if submitted and name and personality and background and speaking_style and knowledge_domains:
            # Create new persona
            domains_list = [domain.strip() for domain in knowledge_domains.split(',')]
            new_persona = AIPersona(
                name=name,
                personality=personality,
                background=background,
                speaking_style=speaking_style,
                knowledge_domains=domains_list
            )
            
            # Add to session state
            st.session_state.personas[name] = new_persona
            
            # Save to file
            save_personas(st.session_state.personas)
            
            st.success(f"Created new persona: {name}")
            return new_persona
        elif submitted:
            st.error("Please fill in all fields")
            return None

def main():
    st.title("🤖 AI Persona Creator")
    
    # Initialize OpenAI API key input
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # Load existing personas
    st.session_state.personas = load_personas()
    
    # Create tabs for different functions
    tab1, tab2 = st.tabs(["Create Persona", "Chat with Persona"])
    
    with tab1:
        new_persona = create_persona_form()
    
    with tab2:
        if st.session_state.personas:
            selected_persona_name = st.selectbox(
                "Select a persona to chat with",
                options=list(st.session_state.personas.keys())
            )
            
            if selected_persona_name:
                st.session_state.current_persona = st.session_state.personas[selected_persona_name]
                
                # Chat interface
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
                
                if prompt := st.chat_input("Type your message..."):
                    # Add user message to chat history
                    st.session_state.messages.append(create_chat_message("user", prompt))
                    
                    # Get AI response
                    if openai_api_key:
                        response = get_ai_response(
                            st.session_state.messages,
                            st.session_state.current_persona
                        )
                        if response:
                            # Add AI response to chat history
                            st.session_state.messages.append(
                                create_chat_message("assistant", response)
                            )
                            st.rerun()
                    else:
                        st.error("Please enter your OpenAI API key in the sidebar")
        else:
            st.info("Please create a persona first")

if __name__ == "__main__":
    main()