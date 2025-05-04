# Angel One RAG Chatbot - Setup and Usage Documentation

## Overview

The Angel One RAG Chatbot is a customer support chatbot that leverages Retrieval-Augmented Generation (RAG) to answer questions about Angel One's services. The chatbot retrieves information from Angel One's support documentation and responds to user queries with relevant, accurate information.

**Live Demo**: [https://angel-one-tag.onrender.com/](https://angel-one-tag.onrender.com/)

**GitHub Repository**: [https://github.com/shyamww/angel_one_tag](https://github.com/shyamww/angel_one_tag)

## Features

- **RAG Technology**: Combines vector similarity search with language generation
- **Multiple Data Sources**: Uses Angel One's support website and insurance PDFs
- **Hybrid Search**: Combines vector embeddings with keyword matching for better results
- **User-Friendly Interface**: Clean, responsive chat interface
- **"I Don't Know" Responses**: Honestly admits when it can't answer a question
- **OpenAI Integration**: Optional integration for better response quality

## System Requirements

- Python 3.8+
- 4GB+ RAM (for local development with full RAG capabilities)
- Internet connection (for initial scraping)

## Setup Instructions

### Local Development Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/shyamww/angel_one_tag.git
   cd angel_one_tag
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Apply Migrations**
   ```bash
   python manage.py migrate
   ```

5. **Run the Development Server**
   ```bash
   python manage.py runserver
   ```

6. **Access the Application**
   - Open your browser and go to http://127.0.0.1:8000/

### Building the Knowledge Base

The repository includes a pre-built knowledge base in the `data/vector_store/` directory. If you want to rebuild it:

1. **Ensure PDF Files Are Available**
   - Place the insurance PDFs in the `data/raw/pdfs/` directory

2. **Build the Knowledge Base**
   - Go to the application homepage
   - Click "Build Knowledge Base"
   - Wait for the process to complete (this may take several minutes)

3. **Alternative: Command Line Build**
   ```bash
   python manage.py shell
   ```
   ```python
   from chatbot.document_processor import DocumentProcessor
   from chatbot.rag_engine import RAGEngine
   import os
   from django.conf import settings
   
   DATA_DIR = os.path.join(settings.BASE_DIR, 'data')
   ANGEL_ONE_URL = "https://www.angelone.in/support"
   CONTENT_SELECTORS = ["main", ".content", "article", ".support-content", ".faq-content", ".knowledge-base", "p"]
   BLACKLIST_URLS = ["/login", "/register", "/download", "/app", "/careers"]
   
   # Process documents
   doc_processor = DocumentProcessor(
       base_url=ANGEL_ONE_URL,
       content_selectors=CONTENT_SELECTORS,
       data_dir=DATA_DIR,
       blacklist_urls=BLACKLIST_URLS
   )
   chunks = doc_processor.process_all()
   
   # Build vector store
   rag_engine = RAGEngine(data_dir=DATA_DIR)
   rag_engine.build_vector_store(chunks)
   
   print(f"Successfully processed {len(chunks)} document chunks")
   ```

## Usage Guide

### Starting a Chat Session

1. Go to the application homepage
2. Click "Start Chatting"
3. Type your question in the input field
4. Press Enter or click the send button
5. The chatbot will respond with relevant information

### Sample Questions

Try asking questions like:
- "How do I open an account?"
- "How to track application status?"
- "What documents are needed for KYC?"
- "How to add a bank account?"
- "What is the process for withdrawing funds?"

### Understanding Responses

The chatbot responses include:
- **Information**: Relevant information from Angel One's documentation
- **Sources**: References to the source documents
- **"I Don't Know" Responses**: If the question is outside the scope of the knowledge base

### Using OpenAI Integration (Optional)

For better quality responses, you can integrate with OpenAI:

1. **Create a .env File**
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

2. **Restart the Server**
   ```bash
   python manage.py runserver
   ```

When an OpenAI API key is provided, the system will use OpenAI's models for more natural, conversational responses.

## Deployment

### Deployment to Render.com

The application is currently deployed on Render.com's free tier:

1. **Create a Render Account**
   - Sign up at [render.com](https://render.com)

2. **Create a New Web Service**
   - Connect your GitHub repository
   - Configure the build settings:
     - Build Command: `pip install -r requirements.txt && python manage.py collectstatic --noinput && python manage.py migrate`
     - Start Command: `gunicorn angel_one_rag.wsgi:application --timeout 300 --workers 1 --threads 1 --preload`

3. **Set Environment Variables**
   - `DEBUG`: False
   - `ALLOWED_HOSTS`: .onrender.com
   - `WEB_CONCURRENCY`: 1
   - `SECRET_KEY`: (generate a secure key)

4. **Deploy the Service**
   - Render will automatically build and deploy the application

### Note About 502 Errors

The deployed application occasionally returns 502 errors due to limitations of Render's free tier:

- **Memory Constraints**: The free tier has limited memory (512MB RAM), which can be insufficient for the vector embedding operations
- **Timeouts**: The free tier has strict timeout limits
- **Cold Starts**: After periods of inactivity, the service may take time to restart

For a production-grade deployment, consider:
- Upgrading to a paid tier on Render
- Deploying to a more powerful platform (AWS, GCP, Azure)
- Using the lightweight version for demonstration and the full version for production

## Architecture

### Components

1. **Document Processor**
   - Scrapes content from Angel One's support website
   - Processes PDF documents
   - Chunks content for better retrieval

2. **RAG Engine**
   - Creates and manages vector embeddings
   - Implements hybrid search (vector + keyword)
   - Generates responses from retrieved content

3. **Web Interface**
   - Django-based web application
   - Chat interface with conversation history
   - Build knowledge base functionality

### File Structure

```
angel_one_tag/
├── angel_one_rag/         # Django project settings
├── chatbot/               # Main application
│   ├── document_processor.py  # Content processing
│   ├── rag_engine.py      # RAG implementation
│   ├── rag_engine_light.py # Lightweight version for deployment
│   ├── views.py           # Web views
│   ├── models.py          # Data models
│   └── templates/         # HTML templates
├── data/                  # Data storage
│   ├── raw/               # Raw text and PDFs
│   │   └── pdfs/          # PDF documents
│   ├── processed/         # Processed chunks
│   └── vector_store/      # FAISS index and metadata
├── requirements.txt       # Dependencies
├── runtime.txt            # Python version
├── Procfile               # Deployment configuration
└── render.yaml            # Render.com configuration
```

## Troubleshooting

### Common Issues

1. **"I don't know" responses for valid questions**
   - Ensure the knowledge base is built
   - Check that your question relates to Angel One's services
   - Try rephrasing the question

2. **Memory errors during knowledge base building**
   - Ensure your system has sufficient RAM (4GB+ recommended)
   - Reduce chunk size in `document_processor.py`
   - Process fewer documents initially

3. **502 errors on the deployed application**
   - This is expected on Render's free tier due to memory constraints
   - Try refreshing the page
   - Consider upgrading to a paid tier for better reliability

4. **Slow response times**
   - The first query after startup may be slow as models load
   - Subsequent queries should be faster
   - The deployed version uses a lightweight implementation for better performance

## Contact and Support

For questions, issues, or contributions, please:
- Create an issue on the GitHub repository
- Contact the project maintainer at [your email]

## License

This project is licensed under the MIT License - see the LICENSE file for details.