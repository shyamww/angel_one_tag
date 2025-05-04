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

        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Vector store
        self.vector_store = None

        # Initialize LLM
        self._init_llm()

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
            max_new_tokens=512,
            truncation=True,
            temperature=0.1
        )

    def build_vector_store(self, chunks: List[Dict[str, Any]]) -> None:
        """Build and save the vector store from document chunks."""
        docs = [
            Document(
                page_content=chunk["content"],
                metadata={
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "title": chunk.get("title", ""),
                    "topic": chunk.get("topic", "general")
                }
            ) for chunk in chunks
        ]

        # Create vector store
        self.vector_store = FAISS.from_documents(docs, self.embeddings)

        # Save vector store
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(self.vector_store_dir))

        # Save raw chunks for reference
        with open(self.vector_store_dir / "chunks.json", 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

    def load_vector_store(self) -> bool:
        """Load vector store from disk."""
        vector_store_path = self.vector_store_dir / "index.faiss"

        if not vector_store_path.exists():
            logger.warning("Vector store not found. Call build_vector_store first.")
            return False

        try:
            self.vector_store = FAISS.load_local(
                str(self.vector_store_dir),
                self.embeddings
            )
            return True
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False

    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
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

            # Perform vector search with score
            docs_with_scores = self.vector_store.similarity_search_with_score(question, k=k)

            # Check for relevance using score threshold
            # FAISS returns distance (lower is better), so we need a max threshold
            max_distance_threshold = 1.2  # Adjust based on testing

            # Filter by score
            relevant_docs = []
            for doc, score in docs_with_scores:
                if score < max_distance_threshold:
                    relevant_docs.append(doc)

            if not relevant_docs:
                return {
                    "answer": "I don't know.",
                    "sources": [],
                    "has_answer": False
                }

            # Continue with hybrid search for better results
            keyword_docs = []
            try:
                chunks_json_path = self.vector_store_dir / "chunks.json"
                if chunks_json_path.exists():
                    with open(chunks_json_path, 'r', encoding='utf-8') as f:
                        all_chunks = json.load(f)

                    keyword_matches = []

                    for chunk in all_chunks:
                        content = chunk["content"].lower()
                        matches = sum(1 for keyword in keywords if keyword in content)
                        if matches >= 2:
                            keyword_matches.append({
                                "chunk": chunk,
                                "matches": matches
                            })

                    if keyword_matches:
                        keyword_matches.sort(key=lambda x: x["matches"], reverse=True)
                        keyword_docs = [
                            Document(
                                page_content=match["chunk"]["content"],
                                metadata={"source": match["chunk"]["source"]}
                            ) for match in keyword_matches[:5]
                        ]
            except Exception as e:
                pass

            # Combine vector and keyword results, avoiding duplicates
            all_docs = []
            seen_content = set()

            for doc in relevant_docs:  # Use filtered relevant_docs
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    all_docs.append(doc)
                    seen_content.add(content_hash)

            for doc in keyword_docs:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content and len(all_docs) < 10:
                    all_docs.append(doc)
                    seen_content.add(content_hash)

            if not all_docs:
                return {
                    "answer": "I don't know.",
                    "sources": [],
                    "has_answer": False
                }

            # Additional relevance check - calculate semantic similarity
            # If retrieved docs don't pass minimum semantic similarity, return "I don't know"
            min_semantic_sim = 0.3  # Adjust based on testing
            has_relevant_docs = False

            # Check at least one document is reasonably similar to question
            question_embedding = self.embeddings.embed_query(question)

            for doc in all_docs[:2]:  # Check top 2 docs
                doc_embedding = self.embeddings.embed_query(doc.page_content[:500])

                # Calculate cosine similarity
                similarity = self._cosine_similarity(question_embedding, doc_embedding)
                if similarity > min_semantic_sim:
                    has_relevant_docs = True
                    break

            if not has_relevant_docs:
                return {
                    "answer": "I don't know.",
                    "sources": [],
                    "has_answer": False
                }

            # Simply concatenate all content without additional processing or formatting
            all_content = []
            sources = []

            for doc in all_docs:
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