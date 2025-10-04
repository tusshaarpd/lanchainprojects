import streamlit as st
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Set your API keys (use environment variables or Streamlit secrets instead!)
# NEVER hardcode API keys in production
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "your_anthropic_api_key_here")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")

st.set_page_config(page_title="Claude Chat with PDFs", page_icon="📄")
st.title("📄 Chat with your PDFs (LangChain + Anthropic Claude)")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Initialize session state for vectorstore
    if "vectorstore" not in st.session_state or st.session_state.get("uploaded_file_name") != uploaded_file.name:
        with st.spinner("Processing PDF..."):
            try:
                # Save file temporarily
                temp_file_path = "temp.pdf"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.read())
                
                # Step 2: Load & Split
                loader = PyPDFLoader(temp_file_path)
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200
                )
                docs = text_splitter.split_documents(documents)
                
                # Step 3: Embeddings + VectorStore
                embeddings = OpenAIEmbeddings()
                vectorstore = FAISS.from_documents(docs, embeddings)
                
                # Store in session state
                st.session_state.vectorstore = vectorstore
                st.session_state.uploaded_file_name = uploaded_file.name
                
                # Clean up temp file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                
                st.success("PDF processed successfully!")
                
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                st.stop()
    
    # Step 4: Memory
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Step 5: Conversational Retrieval Chain with Claude
    if "qa_chain" not in st.session_state:
        # Use correct model identifier for Claude 3 Sonnet
        # For Claude Sonnet 4.5, use: "claude-sonnet-4-5-20250929"
        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.3)
        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=st.session_state.vectorstore.as_retriever(),
            memory=st.session_state.memory,
            verbose=True
        )
    
    # Chat Interface
    user_input = st.text_input("Ask a question about the PDF:", key="user_input")
    
    if st.button("Send") or (user_input and user_input != st.session_state.get("last_input", "")):
        if user_input:
            st.session_state.last_input = user_input
            
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.qa_chain.run(user_input)
                    st.session_state.chat_history.append(("You", user_input))
                    st.session_state.chat_history.append(("Bot", response))
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
    
    # Display conversation
    if st.session_state.chat_history:
        st.subheader("Conversation")
        for speaker, msg in st.session_state.chat_history:
            if speaker == "You":
                st.markdown(f"**🧑 {speaker}:** {msg}")
            else:
                st.markdown(f"**🤖 {speaker}:** {msg}")
        
        # Add clear conversation button
        if st.button("Clear Conversation"):
            st.session_state.chat_history = []
            st.session_state.memory.clear()
            st.rerun()
else:
    st.info("👆 Please upload a PDF file to get started!")