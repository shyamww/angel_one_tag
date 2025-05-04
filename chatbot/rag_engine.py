import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Use community imports to prevent deprecation warnings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI

from transformers import pipeline

logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(self, data_dir: str, use_openai: bool = False, openai_api_key: Optional[str] = None):
        """Initialize the RAG engine."""
        self.data_dir = Path(data_dir)
        self.vector_store_dir = self.data_dir / "vector_store"

        # Auto-detect if we should use OpenAI based on key presence
        self.use_openai = use_openai and openai_api_key and len(openai_api_key) > 20
        self.openai_api_key = openai_api_key

        # Don't initialize embeddings model right away - do it lazily when needed
        self.embeddings = None

        # Vector store
        self.vector_store = None

        # Initialize LLM
        self._init_llm()

    def _get_embeddings(self):
        """Lazily initialize embeddings model when needed."""
        if self.embeddings is None:
            print("Initializing embeddings model...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",  # Smaller model
                model_kwargs={'device': 'cpu'}
            )
        return self.embeddings

    def _init_llm(self):
        """Initialize the language model."""
        if self.use_openai and self.openai_api_key:
            try:
                # Use OpenAI directly
                self.llm = OpenAI(
                    openai_api_key=self.openai_api_key,
                    model_name="gpt-3.5-turbo-instruct",
                    temperature=0
                )
                print("Using OpenAI for generation")
            except Exception as e:
                print(f"Error initializing OpenAI: {str(e)}. Falling back to local model.")
                self.use_openai = False
                self._init_local_model()
        else:
            self._init_local_model()

    def _init_local_model(self):
        """Initialize the local model as fallback."""
        print("Using local model for generation")
        self.llm = pipeline(
            "text-generation",
            model="facebook/opt-350m",
            max_new_tokens=256,  # Reduced from 512
            truncation=True,
            temperature=0.1
        )

    def build_vector_store(self, chunks: List[Dict[str, Any]]) -> None:
        """Build and save the vector store from document chunks."""
        # Process in smaller batches to save memory
        batch_size = 50
        all_docs = []

        # Get embeddings model when needed
        embeddings = self._get_embeddings()

        print(f"Processing {len(chunks)} chunks in batches of {batch_size}")

        # Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}")

            # Create Document objects for this batch
            batch_docs = [
                Document(
                    page_content=chunk["content"],
                    metadata={
                        "source": chunk["source"],
                        "chunk_id": chunk["chunk_id"],
                        "title": chunk.get("title", ""),
                        "topic": chunk.get("topic", "general")
                    }
                ) for chunk in batch
            ]

            all_docs.extend(batch_docs)

        # Create vector store
        print("Creating FAISS vector store...")
        self.vector_store = FAISS.from_documents(
            all_docs,
            embeddings,
            batch_size=32  # Small batch size for FAISS processing
        )

        # Save vector store
        print("Saving vector store...")
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(self.vector_store_dir))

        # Save raw chunks for reference
        print("Saving chunks.json...")
        with open(self.vector_store_dir / "chunks.json", 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print("Vector store build complete!")

    def load_vector_store(self) -> bool:
        """Load vector store from disk."""
        vector_store_path = self.vector_store_dir / "index.faiss"

        if not vector_store_path.exists():
            logger.warning("Vector store not found. Call build_vector_store first.")
            return False

        try:
            # Get embeddings only when needed
            embeddings = self._get_embeddings()

            print("Loading vector store from disk...")
            self.vector_store = FAISS.load_local(
                str(self.vector_store_dir),
                embeddings
            )
            print("Vector store loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False

    def query(self, question: str, k: int = 3) -> Dict[str, Any]:  # Reduced from 5 to 3
        """Query the RAG system and return raw relevant information or 'I don't know'."""
        if not self.vector_store:
            if not self.load_vector_store():
                return {
                    "answer": "I don't know.",
                    "sources": [],
                    "has_answer": False
                }

        try:
            # Extract keywords for search
            stop_words = {'a', 'an', 'the', 'is', 'are', 'what', 'how', 'do', 'does', 'why',
                          'when', 'where', 'which', 'to', 'from', 'in', 'on', 'at', 'by', 'for',
                          'with', 'about', 'as', 'of', 'and', 'or', 'not', 'can', 'you', 'your',
                          'i', 'my', 'me', 'mine', 'we', 'us', 'our'}

            keywords = [word for word in question.lower().split()
                        if len(word) > 3 and word not in stop_words]

            # Simple search to save memory
            print("Performing vector search...")
            docs = self.vector_store.similarity_search(question, k=k)

            if not docs:
                return {
                    "answer": "I don't know.",
                    "sources": [],
                    "has_answer": False
                }

            # Simply concatenate all content without additional processing
            all_content = []
            sources = []

            for doc in docs:
                if doc.page_content.strip():
                    all_content.append(doc.page_content.strip())
                    sources.append(doc.metadata.get("source", "Unknown"))

            # Join with a clear separator
            raw_content = "\n\n-----\n\n".join(all_content)

            # Don't add any formatting like "Based on Angel One support information"
            answer = raw_content

            return {
                "answer": answer,
                "sources": sources,
                "has_answer": bool(all_content)
            }

        except Exception as e:
            import traceback
            print(f"Error in query: {str(e)}")
            print(traceback.format_exc())
            return {
                "answer": "I don't know.",
                "sources": [],
                "has_answer": False
            }

    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = sum(a * a for a in vec1) ** 0.5
        norm_b = sum(b * b for b in vec2) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0

        return dot_product / (norm_a * norm_b)