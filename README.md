# Angel One RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Django that provides customer support by answering queries using Angel One's support documentation.

## Overview

This project implements a RAG chatbot that:
- Answers questions based only on information from Angel One's support documentation
- Responds with "I don't know" for questions outside the scope of the knowledge base
- Features a user-friendly chat interface

The system uses vector embeddings to efficiently retrieve relevant information from a knowledge base built from:
1. Angel One's support website (https://www.angelone.in/support)
2. Various insurance PDFs

## Features

- **Vector-Based Retrieval**: Uses FAISS for efficient similarity search
- **Hybrid Search**: Combines vector search with keyword matching for better results
- **Relevance Filtering**: Smart filtering to ensure high-quality, relevant responses
- **OpenAI Integration**: Optional integration with OpenAI for enhanced response generation
- **User-Friendly Interface**: Clean chat interface with conversation history
- **Fallback Mechanism**: Responds with "I don't know" for out-of-scope questions

## Installation

### Prerequisites
- Python 3.8+
- Django 4.2+
- Internet connection (for initial scraping)
- Angel One support documentation PDFs (included in repository)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/shyamww/angel-one-rag.git
cd angel-one-rag
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Apply Django migrations:
```bash
python manage.py migrate
```

5. (Optional) Create a `.env` file for OpenAI integration:
```
OPENAI_API_KEY=your-api-key-here
```

## Usage

1. Start the Django development server:
```bash
python manage.py runserver
```

2. Open your browser and navigate to http://127.0.0.1:8000/

3. Click "Build Knowledge Base" to create the vector store
   - This will scrape the Angel One support website
   - Process the provided PDF documents
   - Create embeddings for efficient retrieval
   - This process may take several minutes depending on your system

4. Once the knowledge base is built, click "Start Chatting" to begin using the chatbot

5. Ask questions related to Angel One's services, such as:
   - "How do I open an account?"
   - "How to track application status?"
   - "What documents are needed for KYC?"

## Response Modes

The chatbot has two response modes depending on whether an OpenAI API key is provided:

### Local Model Mode (Default)
- Uses local models for generating responses
- Returns raw, relevant information from the knowledge base
- Less conversational but provides comprehensive information
- No API costs involved

### OpenAI Mode (Optional)
- Activated by providing an OpenAI API key in the `.env` file
- Generates more natural, conversational responses
- Provides more focused answers to questions
- Requires an OpenAI API key and incurs API usage costs

## Technical Architecture

### Components

1. **Document Processor**
   - Scrapes content from Angel One's support website
   - Processes PDF documents
   - Chunks content into manageable pieces
   - Adds metadata for better retrieval

2. **RAG Engine**
   - Creates and manages vector embeddings
   - Implements hybrid search (vector + keyword)
   - Handles relevance filtering and scoring
   - Generates responses based on retrieved content

3. **Web Interface**
   - Provides a user-friendly chat interface
   - Displays conversation history
   - Shows source attribution for responses

### Directory Structure

```
angel_one_rag/
├── manage.py
├── requirements.txt
├── .env (optional)
├── angel_one_rag/         # Django project settings
├── chatbot/               # Main application
│   ├── document_processor.py
│   ├── rag_engine.py
│   ├── views.py
│   ├── models.py
│   └── templates/
├── data/                  # Data storage
│   ├── raw/               # Raw text and PDFs
│   ├── processed/         # Processed chunks
│   └── vector_store/      # FAISS index and metadata
```

## Customization

### Adjusting Retrieval Parameters

To modify how the system retrieves information, edit `rag_engine.py`:

- Change `k` value (number of documents retrieved)
- Adjust similarity thresholds
- Modify relevance scoring algorithms

### Adding More Content

To expand the knowledge base:
1. Add more PDFs to the `data/raw/pdfs/` directory
2. Update the base URL in `document_processor.py` if needed
3. Rebuild the knowledge base through the web interface

## Deployment

For deployment to a production environment:

1. Update `settings.py` with appropriate production settings
2. Use a production WSGI server like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn angel_one_rag.wsgi:application
   ```
3. Set up a reverse proxy like Nginx
4. Configure static files serving

## Limitations

- The chatbot can only answer questions based on the provided documentation
- Response quality depends on the content in the knowledge base
- Without OpenAI integration, responses are less conversational
- The system may struggle with complex, multi-part questions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Angel One for providing comprehensive support documentation
- The LangChain community for vector store implementations
- Hugging Face for embedding models