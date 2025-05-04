import os
import json
import uuid
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

from .document_processor import DocumentProcessor
from .rag_engine import RAGEngine
from .models import ChatSession, ChatMessage
from .rag_engine_light import SimpleRAGEngine


# Initialize document processor and RAG engine
DATA_DIR = os.path.join(settings.BASE_DIR, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
VECTOR_STORE_DIR = os.path.join(DATA_DIR, 'vector_store')

ANGEL_ONE_URL = "https://www.angelone.in/support"
CONTENT_SELECTORS = [
    "main",
    ".content",
    "article",
    ".support-content",
    ".faq-content",
    ".knowledge-base",
    "p"
]

BLACKLIST_URLS = [
    "/login",
    "/register",
    "/download",
    "/app",
    "/careers"
]


def index(request):
    # Generate a unique session ID if not already present
    session_id = request.session.get('chat_session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session['chat_session_id'] = session_id

        # Create a new chat session
        ChatSession.objects.get_or_create(session_id=session_id)

    return render(request, 'chatbot/index.html', {'session_id': session_id})


def chat(request):
    session_id = request.session.get('chat_session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session['chat_session_id'] = session_id

        # Create a new chat session
        ChatSession.objects.get_or_create(session_id=session_id)

    # Get previous messages
    try:
        chat_session = ChatSession.objects.get(session_id=session_id)
        messages = chat_session.messages.all()
    except ChatSession.DoesNotExist:
        messages = []

    return render(request, 'chatbot/chat.html', {
        'session_id': session_id,
        'messages': messages
    })


@csrf_exempt
def ask(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        data = json.loads(request.body)
        question = data.get('question')
        session_id = data.get('session_id')

        if not question:
            return JsonResponse({'error': 'Question is required'}, status=400)

        if not session_id:
            session_id = str(uuid.uuid4())

        # Get or create chat session
        chat_session, _ = ChatSession.objects.get_or_create(session_id=session_id)

        # Save user message
        ChatMessage.objects.create(
            session=chat_session,
            role='user',
            content=question
        )

        # Initialize RAG engine - will automatically check if OpenAI key is valid
        import os
        openai_api_key = os.getenv('OPENAI_API_KEY')
        # rag_engine = RAGEngine(
        #     data_dir=DATA_DIR,
        #     use_openai=True,  # Try to use OpenAI if key is available
        #     openai_api_key=openai_api_key
        # )
        rag_engine = SimpleRAGEngine(data_dir=DATA_DIR)

        # Get answer
        result = rag_engine.query(question)

        # Save assistant message
        ChatMessage.objects.create(
            session=chat_session,
            role='assistant',
            content=result['answer']
        )

        return JsonResponse({
            'answer': result['answer'],
            'sources': result['sources'],
            'has_answer': result['has_answer']
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def build_index(request):
    """Check if index exists."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        # Check if chunks file exists
        chunks_path = os.path.join(DATA_DIR, 'vector_store', 'chunks.json')

        if os.path.exists(chunks_path):
            # Load existing chunks to count them
            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)

            return JsonResponse({
                'success': True,
                'message': f'Using existing knowledge base with {len(chunks)} document chunks',
                'chunks_count': len(chunks)
            })
        else:
            return JsonResponse({
                'success': False,
                'error': 'Knowledge base not found. Please build locally and push to repository.'
            }, status=404)

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)