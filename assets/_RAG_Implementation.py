# pages/2_🔍_RAG_Implementation.py
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, PyPDFLoader
import tempfile
import os

st.set_page_config(
    page_title="AI Toolkit",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None

def process_file(uploaded_file):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

    # Load the document
    if uploaded_file.type == "application/pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    
    documents = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    splits = text_splitter.split_documents(documents)
    
    # Create embeddings and vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # Clean up temporary file
    os.unlink(file_path)
    
    return vectorstore

def setup_qa_chain(vectorstore):
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return qa_chain

def main():
    initialize_session_state()
    
    st.title("RAG Implementation")
    
    # Sidebar for OpenAI API key
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # Main content
    st.markdown("""
    This page demonstrates a Retrieval-Augmented Generation (RAG) implementation
    using LangChain and OpenAI. Upload your documents and ask questions about them.
    """)
    
    # File upload section
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files and openai_api_key:
        with st.spinner("Processing documents..."):
            # Process each file and combine into single vectorstore
            vectorstores = []
            for file in uploaded_files:
                vectorstore = process_file(file)
                vectorstores.append(vectorstore)
            
            if len(vectorstores) > 1:
                # Merge vectorstores
                st.session_state.vectorstore = vectorstores[0].merge_from(vectorstores[1:])
            else:
                st.session_state.vectorstore = vectorstores[0]
            
            # Setup QA chain
            st.session_state.qa_chain = setup_qa_chain(st.session_state.vectorstore)
            
        st.success("Documents processed successfully!")
    
    # Question answering section
    st.header("2. Ask Questions")
    
    if st.session_state.qa_chain is not None:
        question = st.text_input("Enter your question:")
        
        if question:
            with st.spinner("Generating answer..."):
                response = st.session_state.qa_chain.run(question)
                
                st.markdown("### Answer")
                st.write(response)
                
                # Optional: Show relevant sources
                st.markdown("### Relevant Sources")
                docs = st.session_state.vectorstore.similarity_search(question, k=2)
                for i, doc in enumerate(docs):
                    with st.expander(f"Source {i+1}"):
                        st.write(doc.page_content)
    
    else:
        st.info("Please upload documents and provide OpenAI API key to start asking questions.")
    
    # Additional information
    with st.expander("How it works"):
        st.markdown("""
        1. **Document Processing**: Documents are split into chunks and converted to embeddings
        2. **Vector Storage**: Embeddings are stored in a FAISS vector store
        3. **Retrieval**: When you ask a question, relevant chunks are retrieved
        4. **Generation**: OpenAI's model generates an answer based on the retrieved context
        """)

if __name__ == "__main__":
    main()