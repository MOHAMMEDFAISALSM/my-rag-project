# 🤖 Faisal's AI Assistant - Advanced RAG Chatbot

A professional Retrieval-Augmented Generation (RAG) chatbot with a stunning Streamlit web interface. Features intelligent document analysis, session-based isolation, and a premium glassmorphism UI.

## ✨ Features

- **📄 PDF Document Upload** - Upload and process multiple PDF files
- **🧠 Intelligent Question Detection** - Automatically detects general vs. document-specific questions
- **💬 Session-Based Isolation** - Each user gets a private workspace
- **🎨 Premium UI** - Glassmorphism design with advanced animations
- **📊 Real-Time Stats** - Track chunks and memory usage
- **💾 Export Chat History** - Download conversations as text files
- **🔍 Source Citations** - View exact pages where answers come from
- **⚡ Suggested Questions** - Quick-start buttons for onboarding

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google Gemini API Key ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd my-rag-project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

5. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

6. **Open your browser**
   
   Navigate to `http://localhost:8502`

## 📁 Project Structure

```
my-rag-project/
├── streamlit_app.py      # Main Streamlit application
├── src/
│   ├── __init__.py       # Package initializer
│   ├── data_loader.py    # Document loading utilities
│   └── vector_store.py   # FAISS vector store management
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (not in git)
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## 🎯 Usage

1. **Upload Documents**
   - Click "Browse files" in the sidebar
   - Select one or more PDF files
   - Click "Process Documents"

2. **Ask Questions**
   - Use the Quick Start buttons for common queries
   - Or type your own questions in the chat input

3. **View Sources**
   - Expand the "View Sources" section to see citations
   - Each answer shows which pages were used

4. **Manage Session**
   - "Clear History" - Clears chat only
   - "Reset Session" - Clears everything (documents + chat)
   - "Download Chat" - Export conversation

## 🔧 Configuration

### Adjust RAG Parameters

In the sidebar, you can customize:
- **Number of chunks (k)**: How many document chunks to retrieve (1-10)
- **Search Type**: 
  - `similarity` - Most similar chunks
  - `mmr` - Maximum Marginal Relevance (diverse results)

### Change LLM Model

Edit `streamlit_app.py` line ~280:
```python
return ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.7)
```

## 🌐 Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub (make sure `.env` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Add your `GOOGLE_API_KEY` in Streamlit Cloud secrets:
   - Go to App Settings → Secrets
   - Add: `GOOGLE_API_KEY = "your_key_here"`
5. Deploy!

### Environment Variables for Deployment

Make sure to set these in your deployment platform:
- `GOOGLE_API_KEY` - Your Google Gemini API key

## 🛡️ Security

- ✅ API keys stored in `.env` (gitignored)
- ✅ Session-based document isolation
- ✅ Temporary file cleanup
- ✅ No persistent storage of user data

## 🎨 UI Features

- **Animated Title** - Multi-color gradient with floating effect
- **Glassmorphism Cards** - Frosted glass effect on all elements
- **Smooth Animations** - Slide-in messages, hover effects
- **Custom Scrollbar** - Gradient purple-to-teal design
- **Responsive Design** - Works on all screen sizes

## 📦 Dependencies

- `streamlit` - Web framework
- `langchain` - LLM orchestration
- `langchain-google-genai` - Google Gemini integration
- `langchain-huggingface` - Embeddings
- `faiss-cpu` - Vector similarity search
- `pypdf` - PDF processing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Mohammed Faisal**

- LinkedIn: [Your LinkedIn Profile]
- GitHub: [Your GitHub Profile]

## 🙏 Acknowledgments

- Google Gemini for the LLM
- Streamlit for the amazing web framework
- LangChain for RAG orchestration
- HuggingFace for embeddings

---

**Made with ❤️ and ✨ by Faisal**
