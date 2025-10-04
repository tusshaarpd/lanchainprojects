import streamlit as st
import os
import pandas as pd
import tempfile
from typing import Optional, List, Dict
import re
from io import BytesIO

# Updated imports for LangChain
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="LLM Response Evaluator + RAG", 
    page_icon="📄",
    layout="wide"
)

# Initialize session state for API keys
if 'anthropic_api_key' not in st.session_state:
    st.session_state.anthropic_api_key = ''
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ''

# ------------------ SIDEBAR FOR API KEYS ------------------
with st.sidebar:
    st.header("🔑 API Configuration")
    
    # API Key inputs with password field
    anthropic_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=st.session_state.anthropic_api_key,
        help="Enter your Anthropic API key for Claude models"
    )
    
    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.openai_api_key,
        help="Enter your OpenAI API key for embeddings"
    )
    
    # Model selection
    model_choice = st.selectbox(
        "Select LLM Model",
        options=["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "gpt-4", "gpt-3.5-turbo"],
        help="Choose the model for evaluation"
    )
    
    # Temperature setting
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Lower values make output more deterministic"
    )
    
    if st.button("Save API Keys"):
        if anthropic_key:
            st.session_state.anthropic_api_key = anthropic_key
            os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        if openai_key:
            st.session_state.openai_api_key = openai_key
            os.environ["OPENAI_API_KEY"] = openai_key
        st.success("API keys saved!")

# ------------------ MAIN APP ------------------
st.title("📄 LLM Response Evaluation with RAG & Feedback")

st.markdown("""
### How to use this app:
1. **Configure API Keys** in the sidebar (required)
2. **Upload an Excel file** with:
   - `Request` column (user prompts)
   - `Response` column (LLM's responses to evaluate)
3. **Optionally upload PDFs** for context-aware evaluation
4. **Click Evaluate** to get scores and feedback
""")

# Check if API keys are configured
api_keys_configured = False
if model_choice.startswith("claude"):
    api_keys_configured = bool(st.session_state.anthropic_api_key and st.session_state.openai_api_key)
else:
    api_keys_configured = bool(st.session_state.openai_api_key)

if not api_keys_configured:
    st.warning("⚠️ Please configure your API keys in the sidebar first!")

# ------------------ FILE UPLOAD ------------------
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Upload Excel File",
        type=["xlsx", "xls"],
        help="Must contain 'Request' and 'Response' columns"
    )

with col2:
    pdf_files = st.file_uploader(
        "Upload PDF(s) for Context (Optional)",
        type="pdf",
        accept_multiple_files=True,
        help="PDFs will be used to provide context for evaluation"
    )

# ------------------ PROCESS EXCEL ------------------
df = None
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        
        # Check for required columns
        required_cols = ["Request", "Response"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Excel file is missing required columns: {missing_cols}")
            st.info("Your file has columns: " + ", ".join(df.columns))
            df = None
        else:
            st.success(f"✅ Excel loaded successfully! {len(df)} rows found.")
            
            # Display preview
            with st.expander("Preview Data"):
                st.dataframe(df.head(10))
            
            # Display statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                avg_request_len = df["Request"].astype(str).str.len().mean()
                st.metric("Avg Request Length", f"{avg_request_len:.0f} chars")
            with col3:
                avg_response_len = df["Response"].astype(str).str.len().mean()
                st.metric("Avg Response Length", f"{avg_response_len:.0f} chars")
                
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}")
        df = None

# ------------------ PROCESS PDFs ------------------
retriever = None
if pdf_files and api_keys_configured:
    with st.spinner("Processing PDF documents..."):
        all_docs = []
        
        for pdf_file in pdf_files:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.getbuffer())
                tmp_path = tmp_file.name
            
            try:
                # Load PDF
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                all_docs.extend(docs)
                st.info(f"📄 Loaded {pdf_file.name}: {len(docs)} pages")
            except Exception as e:
                st.error(f"Error loading {pdf_file.name}: {str(e)}")
            finally:
                # Clean up temp file
                os.unlink(tmp_path)
        
        if all_docs:
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(all_docs)
            
            st.info(f"📚 Created {len(chunks)} text chunks from {len(all_docs)} pages")
            
            # Create embeddings and vector store
            try:
                embeddings = OpenAIEmbeddings()
                vectorstore = FAISS.from_documents(chunks, embeddings)
                retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
                )
                st.success("✅ Context documents processed successfully!")
            except Exception as e:
                st.error(f"Error creating embeddings: {str(e)}")
                retriever = None

# ------------------ EVALUATION FUNCTION ------------------
def extract_score(text: str) -> str:
    """Extract numerical score from evaluation text."""
    # Look for patterns like "8/10", "8 out of 10", "Score: 8", etc.
    patterns = [
        r'(\d+(?:\.\d+)?)\s*/\s*10',
        r'(\d+(?:\.\d+)?)\s+out\s+of\s+10',
        r'[Ss]core:\s*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)/10'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            score = float(match.group(1))
            # Ensure score is within 0-10 range
            if 0 <= score <= 10:
                return str(score)
    
    # If no clear score found, look for any number between 0-10
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
    for num in numbers:
        if 0 <= float(num) <= 10:
            return num
    
    return "N/A"

def evaluate_response(
    llm,
    request: str,
    response: str,
    context: str = "No specific context provided.",
    idx: int = 0,
    total: int = 1
) -> Dict[str, str]:
    """Evaluate a single response using the LLM."""
    
    prompt_template = """You are an expert evaluator of AI responses.

Evaluate the following AI response based on these criteria:
1. **Accuracy**: Is the information correct and factual?
2. **Relevance**: Does it directly address the request?
3. **Completeness**: Does it fully answer the question?
4. **Clarity**: Is it well-written and easy to understand?
5. **Context Alignment**: If context is provided, does it align with it?

Provide your evaluation in the following format:
Score: [X]/10
Summary: [One sentence summary of strengths and weaknesses]

Detailed Feedback:
[2-3 sentences explaining the score, highlighting specific strengths and areas for improvement]

Context Information:
{context}

User Request:
{request}

AI Response to Evaluate:
{response}

Your Evaluation:"""

    try:
        chat_prompt = ChatPromptTemplate.from_template(prompt_template)
        formatted_prompt = chat_prompt.format_prompt(
            context=context,
            request=request,
            response=response
        )
        
        # Progress indicator
        progress_text = f"Evaluating row {idx + 1}/{total}..."
        st.text(progress_text)
        
        # Get evaluation from LLM
        if isinstance(llm, ChatAnthropic):
            result = llm.invoke(formatted_prompt.to_messages()).content
        else:  # ChatOpenAI
            result = llm.invoke(formatted_prompt.to_messages()).content
        
        # Extract score
        score = extract_score(result)
        
        return {
            "feedback": result,
            "score": score
        }
        
    except Exception as e:
        return {
            "feedback": f"Error during evaluation: {str(e)}",
            "score": "Error"
        }

# ------------------ MAIN EVALUATION ------------------
if df is not None and api_keys_configured:
    st.markdown("---")
    st.header("🎯 Evaluation Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=len(df),
            value=min(10, len(df)),
            help="Number of rows to evaluate at once"
        )
    
    with col2:
        use_context = st.checkbox(
            "Use PDF Context for Evaluation",
            value=retriever is not None,
            disabled=retriever is None
        )
    
    if st.button("🚀 Evaluate Responses", type="primary", disabled=not api_keys_configured):
        # Initialize LLM based on selection
        if model_choice.startswith("claude"):
            llm = ChatAnthropic(
                model=model_choice,
                temperature=temperature,
                max_tokens=1000
            )
        else:
            llm = ChatOpenAI(
                model=model_choice,
                temperature=temperature,
                max_tokens=1000
            )
        
        # Initialize results columns
        feedback_list = []
        scores_list = []
        context_list = []
        
        # Create RAG chain if context is available
        qa_chain = None
        if use_context and retriever:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm,
                retriever=retriever,
                memory=memory,
                verbose=False
            )
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Evaluate in batches
        total_rows = min(batch_size, len(df))
        
        for idx in range(total_rows):
            row = df.iloc[idx]
            request_text = str(row["Request"])
            response_text = str(row["Response"])
            
            # Get context if available
            context_snippet = "No specific context provided."
            if qa_chain:
                try:
                    # Retrieve relevant context
                    relevant_docs = retriever.get_relevant_documents(request_text)
                    context_snippet = "\n".join([doc.page_content[:500] for doc in relevant_docs[:2]])
                except Exception as e:
                    context_snippet = f"Context retrieval error: {str(e)}"
            
            context_list.append(context_snippet)
            
            # Evaluate the response
            status_text.text(f"Evaluating row {idx + 1}/{total_rows}...")
            evaluation = evaluate_response(
                llm, 
                request_text, 
                response_text, 
                context_snippet,
                idx,
                total_rows
            )
            
            feedback_list.append(evaluation["feedback"])
            scores_list.append(evaluation["score"])
            
            # Update progress
            progress_bar.progress((idx + 1) / total_rows)
        
        # Add results to dataframe
        results_df = df.head(total_rows).copy()
        results_df["Evaluation_Feedback"] = feedback_list
        results_df["Score"] = scores_list
        if use_context:
            results_df["Context_Used"] = context_list
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Evaluation complete! Evaluated {total_rows} responses.")
        
        # Display results
        st.markdown("### 📊 Evaluation Results")
        
        # Summary statistics
        valid_scores = [s for s in scores_list if s not in ["N/A", "Error"]]
        if valid_scores:
            avg_score = sum(float(s) for s in valid_scores) / len(valid_scores)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Score", f"{avg_score:.2f}/10")
            with col2:
                st.metric("Responses Evaluated", total_rows)
            with col3:
                st.metric("Successful Evaluations", len(valid_scores))
        
        # Display results table
        st.dataframe(
            results_df[["Request", "Response", "Score", "Evaluation_Feedback"]],
            use_container_width=True,
            height=400
        )
        
        # Download results
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            results_df.to_excel(writer, index=False, sheet_name='Evaluations')
        
        st.download_button(
            label="📥 Download Evaluation Results",
            data=output.getvalue(),
            file_name="llm_evaluation_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9em;">
    LLM Response Evaluator with RAG | Built with Streamlit & LangChain
</div>
""", unsafe_allow_html=True)