import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory

# --- Imports are simple now (no re-ranker) ---

from src.data_loader import load_documents
from src.vector_store import get_embeddings, build_vector_store, load_vector_store

# Load environment variables (will look for GOOGLE_API_KEY)
load_dotenv()

def get_llm():
    """Initializes the Gemini LLM."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7 
    )

def create_history_aware_retriever(llm, retriever):
    """
    Creates a chain that rephrases a follow-up question 
    into a standalone question using chat history.
    """
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    return prompt | llm | StrOutputParser()

def create_document_chain(llm):
    """
    Creates the main RAG chain that answers a question 
    based on the retrieved context.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use markdown for formatting, especially bullet points, to make the answer easy to read.

Context: 
{context}
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    return prompt | llm | StrOutputParser()

# --- Main execution ---
if __name__ == "__main__":
    
    # Initialize embeddings
    embeddings = get_embeddings()
    
    # Check if the vector store already exists
    if not os.path.exists("faiss_store"):
        print("No vector store found. Building a new one...")
        # INGESTION PIPELINE
        docs = load_documents("data")
        store = build_vector_store(docs, embeddings)
    else:
        # LOAD EXISTING STORE 
        print("Loading existing vector store.")
        store = load_vector_store(embeddings)

    if store:
        # RETRIEVAL PIPELINE 
        print("Setting up RAG chain...")
        
        # Initialize the LLM
        llm = get_llm()
        
        # --- Default Retriever Settings ---
        retriever_search_type = "similarity"
        retriever_k = 3
        
        # --- Initialize Chat History ---
        chat_history = ChatMessageHistory()
        
        print(f"RAG system ready (using Gemini Flash with Memory).")
        print(f"Current Settings: search_type='{retriever_search_type}', k={retriever_k}")
        print("Type 'set k=5' or 'set search=mmr' to change settings. Type 'exit' to quit.")
        
        while True:
            query = input("You: ")
            if query.lower() == 'exit':
                break
            
            # --- Check for settings commands ---
            if query.lower().startswith("set k="):
                try:
                    new_k = int(query.split("=")[1])
                    if new_k > 0:
                        retriever_k = new_k
                        print(f"-> Settings updated: k={retriever_k}")
                    else:
                        print("-> k must be greater than 0")
                except:
                    print("-> Invalid value. Use 'set k=5', 'set k=3', etc.")
                continue 
                
            if query.lower().startswith("set search="):
                new_type = query.split("=")[1].strip().lower()
                if new_type in ["similarity", "mmr"]:
                    retriever_search_type = new_type
                    print(f"-> Settings updated: search_type='{retriever_search_type}'")
                else:
                    print("-> Invalid type. Use 'set search=similarity' or 'set search=mmr'")
                continue 

            # --- Re-create the retriever with current settings ---
            retriever = store.as_retriever(
                search_type=retriever_search_type,
                search_kwargs={"k": retriever_k}
            )
            
            # --- Re-build the RAG chains ---
            history_aware_retriever_chain = create_history_aware_retriever(llm, retriever)
            document_chain = create_document_chain(llm)
            
            # This is the main runnable that chains everything together
            conversational_rag_chain = (
                RunnableParallel(
                    context=(history_aware_retriever_chain | retriever), 
                    chat_history=lambda x: x["chat_history"],
                    input=lambda x: x["input"]
                )
                | document_chain
            )
            
            print("AI: ", end="", flush=True)
            
            # Use .stream() and collect the full response
            full_response = ""
            for chunk in conversational_rag_chain.stream({
                "input": query,
                "chat_history": chat_history.messages
            }):
                print(chunk, end="", flush=True)
                full_response += chunk
            
            # --- Add conversation to history ---
            chat_history.add_user_message(query)
            chat_history.add_ai_message(full_response)
            
            # Print a final newline
            print("\n")
            
    else:
        print("Could not initialize vector store. Exiting.")