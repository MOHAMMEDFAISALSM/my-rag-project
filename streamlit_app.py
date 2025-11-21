import streamlit as st
import os
import tempfile
import shutil
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory

from src.data_loader import load_documents
from src.vector_store import get_embeddings, build_vector_store, load_vector_store

# Load environment variables
load_dotenv()

# Set API key - works for both local (.env) and Streamlit Cloud (secrets)
if "GOOGLE_API_KEY" not in os.environ:
    try:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except:
        st.error("⚠️ GOOGLE_API_KEY not found! Please add it to Streamlit Cloud Secrets or .env file")
        st.stop()

# --- Page Config ---
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

# --- Advanced Professional CSS with Premium Animations ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* === GLOBAL STYLES === */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-attachment: fixed;
    }

    /* === SIDEBAR === */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(167, 139, 250, 0.2);
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        background: linear-gradient(135deg, #a78bfa 0%, #2dd4bf 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }

    /* === ANIMATED TITLE WITH 3D EFFECT === */
    h1 {
        background: linear-gradient(135deg, #a78bfa 0%, #2dd4bf 50%, #60a5fa 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        text-align: center;
        padding: 30px 0;
        font-size: 3rem;
        letter-spacing: -1px;
        animation: gradientShift 8s ease infinite, titleFloat 3s ease-in-out infinite;
        text-shadow: 0 0 40px rgba(167, 139, 250, 0.3);
        position: relative;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    @keyframes titleFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
    }

    /* === CHAT MESSAGES WITH ADVANCED ANIMATIONS === */
    .stChatMessage {
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.6) 0%, rgba(22, 33, 62, 0.4) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(167, 139, 250, 0.2);
        border-radius: 16px;
        padding: 20px;
        margin: 12px 0;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        animation: messageSlideIn 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stChatMessage::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.03), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes messageSlideIn {
        from { 
            opacity: 0; 
            transform: translateY(20px) scale(0.95);
        }
        to { 
            opacity: 1; 
            transform: translateY(0) scale(1);
        }
    }
    
    @keyframes shimmer {
        to { left: 100%; }
    }
    
    .stChatMessage:hover {
        transform: translateY(-2px);
        box-shadow: 
            0 12px 40px rgba(167, 139, 250, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border-color: rgba(167, 139, 250, 0.4);
    }
    
    [data-testid="stChatMessage"]:nth-child(odd) {
        border-left: 3px solid #a78bfa;
    }

    [data-testid="stChatMessage"]:nth-child(even) {
        border-left: 3px solid #2dd4bf;
    }

    /* === INPUT BOX WITH GLOW === */
    .stChatInput input {
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.8) 0%, rgba(22, 33, 62, 0.6) 100%) !important;
        backdrop-filter: blur(10px);
        color: #e4e4e7 !important;
        border: 2px solid rgba(167, 139, 250, 0.3) !important;
        border-radius: 16px;
        padding: 16px 20px !important;
        font-size: 15px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stChatInput input:focus {
        border-color: #a78bfa !important;
        box-shadow: 
            0 0 0 4px rgba(167, 139, 250, 0.1),
            0 0 30px rgba(167, 139, 250, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        transform: scale(1.01);
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.9) 0%, rgba(22, 33, 62, 0.7) 100%) !important;
    }

    /* === PREMIUM BUTTON STYLING === */
    .stButton > button, 
    button[kind="primary"], 
    button[kind="secondary"],
    [data-testid="baseButton-secondary"],
    [data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, rgba(167, 139, 250, 0.15) 0%, rgba(45, 212, 191, 0.1) 100%) !important;
        backdrop-filter: blur(10px);
        border: 2px solid rgba(167, 139, 250, 0.4) !important;
        color: #e4e4e7 !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        font-weight: 600 !important;
        font-size: 14px;
        letter-spacing: 0.3px;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.1);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(167, 139, 250, 0.25) 0%, rgba(45, 212, 191, 0.15) 100%) !important;
        border-color: #a78bfa !important;
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 
            0 12px 30px rgba(167, 139, 250, 0.3),
            0 0 20px rgba(167, 139, 250, 0.2) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(0.98) !important;
    }

    /* === FILE UPLOADER === */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.4) 0%, rgba(22, 33, 62, 0.3) 100%);
        border: 2px dashed rgba(167, 139, 250, 0.3);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #a78bfa;
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.6) 0%, rgba(22, 33, 62, 0.5) 100%);
        box-shadow: 0 8px 25px rgba(167, 139, 250, 0.2);
        transform: scale(1.01);
    }

    /* === EXPANDER WITH SLIDE EFFECT === */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.6) 0%, rgba(22, 33, 62, 0.4) 100%) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(167, 139, 250, 0.2) !important;
        padding: 12px 16px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.8) 0%, rgba(22, 33, 62, 0.6) 100%) !important;
        border-color: #a78bfa !important;
        transform: translateX(8px);
        box-shadow: -4px 0 15px rgba(167, 139, 250, 0.3);
    }

    /* === STATUS BOX WITH PULSE === */
    [data-testid="stStatusWidget"] {
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.8) 0%, rgba(22, 33, 62, 0.6) 100%) !important;
        border: 2px solid rgba(167, 139, 250, 0.3) !important;
        border-radius: 16px !important;
        padding: 16px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        animation: statusPulse 2s ease-in-out infinite;
    }
    
    @keyframes statusPulse {
        0%, 100% { 
            box-shadow: 0 8px 32px rgba(167, 139, 250, 0.2);
            border-color: rgba(167, 139, 250, 0.3);
        }
        50% { 
            box-shadow: 0 8px 40px rgba(167, 139, 250, 0.4);
            border-color: rgba(167, 139, 250, 0.5);
        }
    }

    /* === METRICS WITH SCALE ANIMATION === */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.6) 0%, rgba(22, 33, 62, 0.4) 100%);
        padding: 20px;
        border-radius: 14px;
        border: 1px solid rgba(167, 139, 250, 0.2);
        animation: metricFadeIn 0.6s ease-out;
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(167, 139, 250, 0.2);
    }
    
    @keyframes metricFadeIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }

    /* === DOWNLOAD BUTTON SPECIAL === */
    [data-testid="stDownloadButton"] button {
        background: linear-gradient(135deg, rgba(45, 212, 191, 0.15) 0%, rgba(167, 139, 250, 0.1) 100%) !important;
        border: 2px solid rgba(45, 212, 191, 0.4) !important;
    }
    
    [data-testid="stDownloadButton"] button:hover {
        border-color: #2dd4bf !important;
        box-shadow: 0 12px 30px rgba(45, 212, 191, 0.3) !important;
    }

    /* === SLIDER STYLING === */
    .stSlider {
        padding: 15px 0;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #a78bfa 0%, #2dd4bf 100%) !important;
    }
    
    .stSlider > div > div > div {
        background: rgba(167, 139, 250, 0.2) !important;
    }
    
    /* === SELECT BOX STYLING === */
    .stSelectbox > div > div {
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.8) 0%, rgba(22, 33, 62, 0.6) 100%) !important;
        border: 2px solid rgba(167, 139, 250, 0.3) !important;
        border-radius: 12px !important;
        color: #e4e4e7 !important;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #a78bfa !important;
        box-shadow: 0 0 15px rgba(167, 139, 250, 0.2);
    }
    
    /* Select dropdown menu */
    [data-baseweb="popover"] {
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.95) 0%, rgba(22, 33, 62, 0.9) 100%) !important;
        backdrop-filter: blur(20px);
        border: 2px solid rgba(167, 139, 250, 0.3) !important;
        border-radius: 12px !important;
    }
    
    [data-baseweb="select"] > div {
        background: transparent !important;
        border-color: rgba(167, 139, 250, 0.3) !important;
    }
    
    /* === USER CHAT MESSAGE SPECIAL STYLING === */
    [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, rgba(167, 139, 250, 0.15) 0%, rgba(45, 212, 191, 0.1) 100%);
        border-radius: 12px;
        padding: 12px 16px;
    }
    
    /* Make user messages stand out more */
    .stChatMessage[data-testid*="user"] {
        background: linear-gradient(135deg, rgba(167, 139, 250, 0.12) 0%, rgba(45, 212, 191, 0.08) 100%);
        border: 2px solid rgba(167, 139, 250, 0.3);
        border-left: 4px solid #a78bfa !important;
    }
    
    /* === DIVIDER === */
    hr {
        border-color: rgba(167, 139, 250, 0.2) !important;
        margin: 24px 0;
    }
    
    /* === SCROLLBAR === */
    * {
        scrollbar-width: thin;
        scrollbar-color: #a78bfa rgba(26, 26, 46, 0.4);
    }
    
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(26, 26, 46, 0.4);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #a78bfa 0%, #2dd4bf 100%);
        border-radius: 10px;
        border: 2px solid rgba(26, 26, 46, 0.4);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #b794ff 0%, #3de4d0 100%);
    }
    
    /* Smooth scrolling for all elements */
    html {
        scroll-behavior: smooth;
    }
    
    * {
        scroll-behavior: smooth;
    }
    
    /* Optimize animations for better performance */
    .stChatMessage, .stButton > button, [data-testid="stMetric"] {
        will-change: transform;
        transform: translateZ(0);
        -webkit-transform: translateZ(0);
    }
    
    /* Reduce motion for better scrolling */
    @media (prefers-reduced-motion: reduce) {
        * {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
    
    /* === INFO BOX STYLING === */
    .stAlert {
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.15) 0%, rgba(167, 139, 250, 0.1) 100%) !important;
        border: 2px solid rgba(96, 165, 250, 0.3) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

st.title("✨ Faisal's AI Assistant")

# --- RAG Chain Functions ---
def create_history_aware_retriever(llm, retriever):
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    return prompt | llm | StrOutputParser()

def create_document_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent assistant that can handle both general questions and document-specific questions.

**IMPORTANT INSTRUCTIONS:**
1. First, analyze if the user's question is asking about the uploaded documents OR if it's a general question (like greetings, general knowledge, etc.)

2. If it's a GENERAL question (greetings, general knowledge, casual conversation):
   - Respond naturally WITHOUT mentioning the documents
   - Do NOT reference the context provided below
   - Be friendly and helpful
   - Examples: "hi", "hello", "what is AI?", "how are you?", "tell me a joke"

3. If it's a DOCUMENT-SPECIFIC question (asking about the content of uploaded files):
   - Use ONLY the retrieved context below to answer
   - Cite specific information from the documents
   - If the context doesn't contain the answer, say "I don't find that information in the uploaded documents."
   - Examples: "summarize the document", "what are the key topics?", "what does the document say about X?"

Retrieved Context (only use if question is about documents):
{context}

Remember: Be smart about detecting the question type. Don't force document context into general conversations."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    return prompt | llm | StrOutputParser()

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history_obj" not in st.session_state:
    st.session_state.chat_history_obj = ChatMessageHistory()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- Caching Resources ---
@st.cache_resource
def get_cached_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

@st.cache_resource
def get_cached_embeddings():
    return get_embeddings()

# --- Sidebar Settings ---
with st.sidebar:
    st.header("Settings")
    
    st.subheader("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        for uploaded_file in uploaded_files:
                            temp_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                        
                        docs = load_documents(temp_dir)
                        embeddings = get_cached_embeddings()
                        st.session_state.vector_store = build_vector_store(docs, embeddings)
                        
                        st.success(f"Processed {len(uploaded_files)} documents!")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error processing documents: {e}")

    if st.button("🗑️ Reset Session", help="Clears all data and chat history"):
        st.session_state.messages = []
        st.session_state.chat_history_obj = ChatMessageHistory()
        st.session_state.vector_store = None
        st.rerun()

    st.divider()
    
    k_value = st.slider("Number of chunks (k)", min_value=1, max_value=10, value=3)
    search_type = st.selectbox("Search Type", ["similarity", "mmr"])
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear History"):
            st.session_state.messages = []
            st.session_state.chat_history_obj = ChatMessageHistory()
            st.rerun()
            
    with col2:
        chat_str = "--- Chat History ---\n\n"
        for msg in st.session_state.messages:
            role = "User" if msg["role"] == "user" else "AI"
            chat_str += f"{role}: {msg['content']}\n\n"
            
        st.download_button(
            label="Download Chat",
            data=chat_str,
            file_name="chat_history.txt",
            mime="text/plain"
        )
        
    if st.session_state.vector_store:
        st.divider()
        st.header("📊 Knowledge Stats")
        num_chunks = st.session_state.vector_store.index.ntotal if hasattr(st.session_state.vector_store, "index") else 0
        st.metric("Chunks", num_chunks)
        st.caption(f"🧠 Memory Size: {num_chunks * 768 / 1024 / 1024:.2f} MB (Est.)")

# --- Initialize chain variables ---
retriever = None
history_aware_retriever_chain = None
document_chain = None

# --- Main Logic ---
try:
    llm = get_cached_llm()
    vector_store = st.session_state.vector_store
    
    if vector_store:
        retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k_value}
        )
        
        history_aware_retriever_chain = create_history_aware_retriever(llm, retriever)
        document_chain = create_document_chain(llm)
    else:
        if len(st.session_state.messages) == 0:
            st.info("👈 Upload a PDF to start chatting!")

except Exception as e:
    st.error(f"An error occurred: {e}")

# --- Chat Interface ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if len(st.session_state.messages) == 0 and st.session_state.vector_store:
    st.markdown("### 💡 Quick Start")
    col1, col2, col3 = st.columns(3)
    
    prompt = None
    if col1.button("📄 Summarize Docs"):
        prompt = "Summarize the uploaded documents in 3 bullet points."
    if col2.button("🔑 Key Topics"):
        prompt = "What are the key topics mentioned in the documents?"
    if col3.button("❓ Common Q&A"):
        prompt = "Generate 3 common questions and answers based on the documents."
else:
    prompt = None

if chat_input := st.chat_input("What is your question?"):
    prompt = chat_input

if prompt and st.session_state.vector_store and retriever and history_aware_retriever_chain and document_chain:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # First, check if this is likely a general question (heuristic)
            general_keywords = ["hi", "hello", "hey", "how are you", "what is", "who is", "tell me about", 
                              "explain", "define", "thanks", "thank you", "bye", "goodbye"]
            is_likely_general = any(keyword in prompt.lower() for keyword in general_keywords)
            
            # Check if question explicitly asks about "document" or "file"
            is_document_question = any(word in prompt.lower() for word in ["document", "file", "pdf", "uploaded", "summarize"])
            
            with st.status("🧠 Thinking...", expanded=True) as status:
                # Always retrieve chunks, but we'll decide whether to use them
                st.write("🔍 Analyzing question...")
                
                retrieval_chain = history_aware_retriever_chain | retriever
                docs = retrieval_chain.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history_obj.messages
                })
                
                # Calculate relevance score (simple keyword matching)
                relevance_score = 0
                prompt_words = set(prompt.lower().split())
                
                for doc in docs:
                    doc_words = set(doc.page_content.lower().split())
                    # Calculate overlap
                    overlap = len(prompt_words.intersection(doc_words))
                    relevance_score += overlap
                
                # Normalize by number of words in prompt
                relevance_score = relevance_score / (len(prompt_words) * len(docs)) if len(prompt_words) > 0 else 0
                
                # Decide whether to use document context
                use_documents = False
                if is_document_question:
                    # User explicitly asked about documents
                    use_documents = True
                    st.write(f"📚 Found {len(docs)} relevant chunks")
                elif relevance_score > 0.3:  # Threshold for relevance
                    # Documents seem relevant
                    use_documents = True
                    st.write(f"📚 Found {len(docs)} relevant chunks")
                else:
                    # Likely a general question with low relevance
                    use_documents = False
                    st.write("💬 Answering as general question")
                
                st.write("✨ Generating answer...")
                status.update(label="✅ Answer Generated", state="complete", expanded=False)
            
            # Generate response based on whether we're using documents
            if use_documents:
                # Use document context
                stream = document_chain.stream({
                    "context": docs,
                    "input": prompt,
                    "chat_history": st.session_state.chat_history_obj.messages
                })
            else:
                # General question - use LLM directly without document context
                general_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a friendly and helpful AI assistant. Answer questions naturally and conversationally."),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("user", "{input}"),
                ])
                general_chain = general_prompt | llm | StrOutputParser()
                stream = general_chain.stream({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history_obj.messages
                })
            
            for chunk in stream:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # Only show sources if we used documents
            if use_documents:
                with st.expander("📚 View Sources"):
                    unique_sources = set()
                    for doc in docs:
                        source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
                        page_num = doc.metadata.get("page", "Unknown")
                        unique_sources.add(f"{source_name} (Page {page_num})")
                    
                    for source in unique_sources:
                        st.markdown(f"- 📄 `{source}`")
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.chat_history_obj.add_user_message(prompt)
            st.session_state.chat_history_obj.add_ai_message(full_response)
            
            if chat_input is None:
                st.rerun()
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
