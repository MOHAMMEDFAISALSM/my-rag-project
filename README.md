# 🧠 RAG Chatbot from Scratch (Gemini + LangChain)

This is a complete **Retrieval-Augmented Generation (RAG)** system built from scratch using **Python**, **LangChain**, and the **Google Gemini 2.5 Flash API**.

The goal of this project was to **learn the fundamentals of RAG** by building and tuning the two core pipelines.  
This application loads PDF documents, processes them into a vector store, and then uses the Gemini LLM to answer questions about the document's content.

---

## 💡 The "Aha!" Moment: Solving Retrieval Failure

This project isn't just about building a RAG pipeline — it's about **understanding how to tune it**.  
Here’s a perfect example of the app failing a query with a small context (`k=3`) and then succeeding after the retriever was tuned to use a wider context (`k=10`).

> This shows the real-world process of finding a failure and engineering a solution.

![Before-and-After tuning the RAG retriever](PASTE YOUR SCREENSHOT URL HERE)

*(This screenshot shows the app failing to find the “list of contributions” with k=3, and then succeeding after running the `set k=10` command).*

---

## ⚙️ Core Features

### 🧩 Data Ingestion Pipeline
- Loads `.pdf`, `.txt`, and `.md` documents from the `/data` folder.  
- Chunks them into smaller pieces.  
- Embeds them into a **FAISS vector store** using **all-MiniLM-L6-v2** embeddings.

### 🔍 Retrieval & Generation Pipeline
- Takes a user's question.  
- Finds the most relevant document chunks.  
- Passes them (along with the question) to the **Gemini 2.5 Flash LLM** to generate an answer.

### 💬 Chat History (Memory)
- The app remembers the previous turns of the conversation.  
- You can ask follow-up questions like *“What are its disadvantages?”* and it will respond contextually.

### 🔧 Dynamic Retriever Tuning
You can **change the retriever’s settings live in the chat!**
```bash
set k=10       # Changes the number of documents to retrieve
set search=mmr # Changes the search algorithm to Max Marginal Relevance
