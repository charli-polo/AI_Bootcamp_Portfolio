# pages/3_🎯_Finetuning.py
import streamlit as st
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import tempfile
import os
import plotly.express as px
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Model Fine-tuning",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'tokenizer' not in st.session_state:
        st.session_state.tokenizer = None
    if 'training_args' not in st.session_state:
        st.session_state.training_args = None
    if 'training_history' not in st.session_state:
        st.session_state.training_history = {'loss': [], 'eval_loss': []}

def load_data(file):
    """Load and validate CSV data"""
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def prepare_dataset(df, text_column, label_column, tokenizer):
    """Prepare dataset for training"""
    # Convert labels to numeric if they're not already
    if df[label_column].dtype == 'object':
        label_map = {label: i for i, label in enumerate(df[label_column].unique())}
        df[label_column] = df[label_column].map(label_map)
        st.session_state.label_map = label_map
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Convert to datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            padding="max_length",
            truncation=True,
            max_length=128
        )
    
    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    return train_dataset, val_dataset

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = np.mean(predictions == labels)
    
    # Calculate precision, recall, f1 for binary classification
    if len(np.unique(labels)) == 2:
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    return {"accuracy": accuracy}

class CustomTrainer(Trainer):
    """Custom trainer class to track training progress"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_history = {'loss': [], 'eval_loss': []}
    
    def log(self, logs):
        super().log(logs)
        if 'loss' in logs:
            self.training_history['loss'].append(logs['loss'])
        if 'eval_loss' in logs:
            self.training_history['eval_loss'].append(logs['eval_loss'])
        st.session_state.training_history = self.training_history

def plot_training_history():
    """Plot training and evaluation loss"""
    if len(st.session_state.training_history['loss']) > 0:
        df = pd.DataFrame({
            'Step': range(len(st.session_state.training_history['loss'])),
            'Training Loss': st.session_state.training_history['loss']
        })
        
        fig = px.line(df, x='Step', y='Training Loss', title='Training Progress')
        st.plotly_chart(fig)

def main():
    initialize_session_state()
    
    st.title("🎯 Fine-tune Hugging Face Models")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Model Configuration")
        model_name = st.selectbox(
            "Select base model",
            ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"]
        )
        num_labels = st.number_input("Number of labels", min_value=2, value=2)
        
        st.header("Training Configuration")
        learning_rate = st.select_slider(
            "Learning rate",
            options=[1e-5, 2e-5, 3e-5, 4e-5, 5e-5],
            value=2e-5
        )
        num_epochs = st.slider("Number of epochs", min_value=1, max_value=10, value=3)
        batch_size = st.select_slider(
            "Batch size",
            options=[8, 16, 32],
            value=16
        )
    
    # Main content
    st.markdown("""
    ### How to use:
    1. Upload your labeled dataset (CSV format)
    2. Configure model and training parameters
    3. Initialize and train the model
    4. Test your fine-tuned model
    """)
    
    # Data upload section
    st.header("1. Upload Training Data")
    uploaded_file = st.file_uploader(
        "Upload your CSV file (should contain text and label columns)",
        type=["csv"]
    )
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Column selection
            col1, col2 = st.columns(2)
            with col1:
                text_column = st.selectbox("Select text column", df.columns)
            with col2:
                label_column = st.selectbox("Select label column", df.columns)
            
            # Model initialization
            if st.button("Initialize Model"):
                with st.spinner("Initializing model and tokenizer..."):
                    try:
                        st.session_state.tokenizer = AutoTokenizer.from_pretrained(model_name)
                        st.session_state.model = AutoModelForSequenceClassification.from_pretrained(
                            model_name,
                            num_labels=num_labels
                        )
                        
                        # Prepare datasets
                        train_dataset, val_dataset = prepare_dataset(
                            df,
                            text_column,
                            label_column,
                            st.session_state.tokenizer
                        )
                        
                        # Setup training arguments
                        output_dir = f"./results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        st.session_state.training_args = TrainingArguments(
                            output_dir=output_dir,
                            learning_rate=learning_rate,
                            num_train_epochs=num_epochs,
                            per_device_train_batch_size=batch_size,
                            per_device_eval_batch_size=batch_size,
                            evaluation_strategy="epoch",
                            save_strategy="epoch",
                            load_best_model_at_end=True,
                            push_to_hub=False,
                            logging_steps=10
                        )
                        
                        # Initialize trainer
                        st.session_state.trainer = CustomTrainer(
                            model=st.session_state.model,
                            args=st.session_state.training_args,
                            train_dataset=train_dataset,
                            eval_dataset=val_dataset,
                            compute_metrics=compute_metrics,
                        )
                        
                        st.success("Model initialized successfully!")
                    except Exception as e:
                        st.error(f"Error initializing model: {str(e)}")
            
            # Training section
            if st.session_state.model is not None:
                st.header("2. Train Model")
                
                if st.button("Start Training"):
                    with st.spinner("Training in progress..."):
                        try:
                            # Create progress placeholder
                            progress_placeholder = st.empty()
                            
                            # Train the model
                            result = st.session_state.trainer.train()
                            
                            # Display training results
                            st.subheader("Training Results")
                            metrics = result.metrics
                            for key, value in metrics.items():
                                st.metric(key, f"{value:.4f}")
                            
                            # Plot training history
                            plot_training_history()
                            
                            # Save model
                            output_dir = "./fine_tuned_model"
                            st.session_state.trainer.save_model(output_dir)
                            st.success(f"Model saved to {output_dir}")
                        
                        except Exception as e:
                            st.error(f"Error during training: {str(e)}")
                
                # Model testing section
                st.header("3. Test Model")
                test_text = st.text_area("Enter text to test:")
                
                if test_text and st.button("Run Inference"):
                    with st.spinner("Generating prediction..."):
                        try:
                            # Tokenize input
                            inputs = st.session_state.tokenizer(
                                test_text,
                                padding=True,
                                truncation=True,
                                max_length=128,
                                return_tensors="pt"
                            )
                            
                            # Get prediction
                            outputs = st.session_state.model(**inputs)
                            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                            predicted_class = torch.argmax(predictions, dim=-1).item()
                            
                            # Display results
                            st.subheader("Prediction Results")
                            
                            # If we have a label map, use it to show the original label
                            if hasattr(st.session_state, 'label_map'):
                                reverse_map = {v: k for k, v in st.session_state.label_map.items()}
                                predicted_label = reverse_map.get(predicted_class, predicted_class)
                                st.write(f"Predicted Class: {predicted_label}")
                            else:
                                st.write(f"Predicted Class: {predicted_class}")
                            
                            # Show probabilities
                            probs = predictions[0].detach().numpy()
                            prob_df = pd.DataFrame({
                                'Class': range(len(probs)),
                                'Probability': probs
                            })
                            st.bar_chart(prob_df.set_index('Class'))
                            
                        except Exception as e:
                            st.error(f"Error during inference: {str(e)}")

if __name__ == "__main__":
    main()