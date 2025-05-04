import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional


class SimpleRAGEngine:
    def __init__(self, data_dir: str):
        """Initialize a lightweight RAG engine."""
        self.data_dir = Path(data_dir)
        self.vector_store_dir = self.data_dir / "vector_store"
        self.chunks = []
        self._load_chunks()

    def _load_chunks(self):
        """Load chunks from JSON file."""
        chunks_path = self.vector_store_dir / "chunks.json"
        if chunks_path.exists():
            try:
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    self.chunks = json.load(f)
                print(f"Loaded {len(self.chunks)} chunks from disk")
            except Exception as e:
                print(f"Error loading chunks: {str(e)}")
                self.chunks = []

    def query(self, question: str, k: int = 3) -> Dict[str, Any]:
        """Query using simple keyword matching."""
        if not self.chunks:
            return {
                "answer": "I don't know.",
                "sources": [],
                "has_answer": False
            }

        # Extract keywords
        keywords = [word.lower() for word in question.split()
                    if len(word) > 3 and word.lower() not in
                    {'what', 'how', 'why', 'when', 'where', 'who', 'which',
                     'and', 'the', 'for', 'with', 'that', 'this'}]

        # Score chunks by keyword matches
        scored_chunks = []
        for chunk in self.chunks:
            content = chunk["content"].lower()
            score = 0
            for keyword in keywords:
                if keyword in content:
                    score += 1

            if score > 0:
                scored_chunks.append((chunk, score))

        # Sort by score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Return top k results
        if not scored_chunks:
            return {
                "answer": "I don't know.",
                "sources": [],
                "has_answer": False
            }

        # Get top chunks
        top_chunks = [chunk for chunk, _ in scored_chunks[:k]]
        sources = [chunk["source"] for chunk in top_chunks]

        # Combine content
        all_content = []
        for chunk in top_chunks:
            if chunk["content"].strip():
                all_content.append(chunk["content"].strip())

        # Join with separator
        raw_content = "\n\n-----\n\n".join(all_content)

        return {
            "answer": raw_content,
            "sources": sources,
            "has_answer": bool(all_content)
        }