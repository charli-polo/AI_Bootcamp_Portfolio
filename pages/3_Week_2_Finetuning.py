import streamlit as st
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import evaluate
import os
from datetime import datetime

st.set_page_config(
    page_title="Gemma 2B Finetuning",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'tokenizer' not in st.session_state:
        st.session_state.tokenizer = None
    if 'training_args' not in st.session_state:
        st.session_state.training_args = None
    if 'trainer' not in st.session_state:
        st.session_state.trainer = None

def load_model_and_tokenizer(model_name):
    """Load the model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit=True
    )
    return model, tokenizer

def prepare_dataset(df, text_column, label_column, tokenizer):
    """Prepare dataset for training"""
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            padding="max_length",
            truncation=True,
            max_length=128
        )
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    return train_dataset, val_dataset

def main():
    st.title("🎯 Finetune Gemma 2B")
    initialize_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Model Configuration")
        model_name = st.selectbox(
            "Select base model",
            ["google/gemma-2b-it", "google/gemma-2b"]
        )
        
        st.header("Training Configuration")
        learning_rate = st.select_slider(
            "Learning rate",
            options=[1e-5, 2e-5, 3e-5, 4e-5, 5e-5],
            value=2e-5
        )
        num_epochs = st.slider("Number of epochs", min_value=1, max_value=10, value=3)
        batch_size = st.select_slider(
            "Batch size",
            options=[4, 8, 16],
            value=8
        )
        
        # Add HuggingFace token input
        hf_token = st.text_input("HuggingFace Token", type="password")
        if hf_token:
            os.environ["HUGGINGFACE_TOKEN"] = hf_token
    
    # Main content
    st.markdown("""
    ### How to use:
    1. Upload your training data (CSV format)
    2. Configure model and training parameters
    3. Start the finetuning process
    4. Evaluate the model
    """)
    
    # Data upload section
    st.header("1. Upload Training Data")
    uploaded_file = st.file_uploader(
        "Upload your CSV file (should contain text and label columns)",
        type=["csv"]
    )
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:", df.head())
        
        # Column selection
        text_column = st.selectbox("Select text column", df.columns)
        label_column = st.selectbox("Select label column", df.columns)
        
        # Initialize model button
        if st.button("Initialize Model"):
            with st.spinner("Loading model and tokenizer..."):
                try:
                    model, tokenizer = load_model_and_tokenizer(model_name)
                    st.session_state.model = model
                    st.session_state.tokenizer = tokenizer
                    st.success("Model and tokenizer loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
        
        # Training section
        if st.session_state.model and st.session_state.tokenizer:
            st.header("2. Start Finetuning")
            
            if st.button("Start Finetuning"):
                with st.spinner("Preparing datasets..."):
                    train_dataset, val_dataset = prepare_dataset(
                        df, text_column, label_column, st.session_state.tokenizer
                    )
                
                # Set up training arguments
                training_args = TrainingArguments(
                    output_dir=f"./results_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    num_train_epochs=num_epochs,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    learning_rate=learning_rate,
                    weight_decay=0.01,
                    logging_dir="./logs",
                    logging_steps=10,
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    load_best_model_at_end=True,
                )
                
                # Initialize trainer
                trainer = Trainer(
                    model=st.session_state.model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                )
                
                # Start training
                with st.spinner("Training in progress..."):
                    try:
                        trainer.train()
                        st.success("Training completed!")
                        
                        # Save the model
                        save_dir = f"./finetuned_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        trainer.save_model(save_dir)
                        st.success(f"Model saved to {save_dir}")
                        
                        # Evaluation metrics
                        eval_results = trainer.evaluate()
                        st.write("Evaluation Results:", eval_results)
                        
                    except Exception as e:
                        st.error(f"Error during training: {str(e)}")
    
    # Add documentation section
    with st.expander("📚 Documentation"):
        st.markdown("""
        ### Model Details
        - **Gemma 2B**: Google's latest language model
        - **Architecture**: Transformer-based
        - **Parameters**: 2 billion
        
        ### Training Parameters
        - **Learning Rate**: Controls how much to adjust the model in response to errors
        - **Batch Size**: Number of samples processed before model update
        - **Epochs**: Number of complete passes through the training dataset
        
        ### Requirements
        - GPU with at least 8GB VRAM
        - HuggingFace account and token
        - Training data in CSV format
        """)

if __name__ == "__main__":
    main()
