import os
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path
import tempfile
from typing import List, Dict, Any, Optional, Tuple
import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain.schema import Document
import shutil

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self, base_url: str, content_selectors: List[str],
                 data_dir: str, blacklist_urls: Optional[List[str]] = None):
        """Initialize the document processor.

        Args:
            base_url: The base URL to scrape.
            content_selectors: CSS selectors to extract content.
            data_dir: Directory to store processed data.
            blacklist_urls: URLs to skip when crawling.
        """
        self.base_url = base_url
        self.content_selectors = content_selectors
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.pdf_dir = self.raw_dir / "pdfs"
        self.processed_dir = self.data_dir / "processed"

        self.blacklist_urls = blacklist_urls or []
        self.visited_urls = set()

        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Smaller chunks for more precision
            chunk_overlap=100,  # Less overlap
            length_function=len,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]  # Better separation
        )

    def is_valid_url(self, url: str) -> bool:
        """Check if a URL is valid and within relevant sections."""
        if not url or not url.startswith('http'):
            return False

        # Filter out telephone and mailto links
        if 'tel:' in url or 'mailto:' in url:
            return False

        parsed_base = urlparse(self.base_url)
        parsed_url = urlparse(url)

        # Check if URL is within same domain
        if parsed_base.netloc != parsed_url.netloc:
            return False

        # Ensure URL is within relevant sections
        # Accept both /support and /knowledge-center paths
        is_support = '/support' in parsed_url.path
        is_knowledge = '/knowledge-center' in parsed_url.path

        if not (is_support or is_knowledge):
            return False

        # Check blacklist
        for blacklisted in self.blacklist_urls:
            if blacklisted in url:
                return False

        return True

    def scrape_page(self, url: str) -> Tuple[str, List[str]]:
        """Scrape a single page and extract content and links."""
        if url in self.visited_urls:
            return "", []

        self.visited_urls.add(url)
        logger.info(f"Scraping {url}")

        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Failed to scrape {url}: {response.status_code}")
                return "", []

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract content using provided selectors
            content = ""
            for selector in self.content_selectors:
                elements = soup.select(selector)
                for element in elements:
                    content += element.get_text(strip=True, separator=" ") + "\n\n"

            # Extract links for further crawling
            links = []
            for a_tag in soup.find_all('a', href=True):
                href = a_tag.get('href')

                # Clean up the href - remove any whitespace and fix malformed URLs
                href = href.strip()

                # Fix malformed URLs
                if ' http' in href:
                    # Split by space and take the first URL only
                    href = href.split(' ')[0]

                # Build the full URL
                full_url = urljoin(url, href)

                if self.is_valid_url(full_url) and full_url not in self.visited_urls:
                    links.append(full_url)

            # Clean content
            content = re.sub(r'\s+', ' ', content).strip()
            content = f"URL: {url}\n\n{content}"

            return content, links

        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return "", []

    def scrape_website(self, start_url: Optional[str] = None) -> List[str]:
        """Crawl the website starting from the base_url or provided start_url."""
        max_pages = 50  # Limit to 50 pages
        scraped_contents = []
        page_count = 0

        to_visit = [start_url or self.base_url]
        scraped_contents = []

        while to_visit and page_count < max_pages:
            current_url = to_visit.pop(0)
            page_count += 1
            content, new_links = self.scrape_page(current_url)

            if content:
                file_name = self._get_filename_from_url(current_url)
                file_path = self.raw_dir / f"{file_name}.txt"

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                scraped_contents.append(str(file_path))

            # Add new links to visit
            for link in new_links:
                if link not in self.visited_urls:
                    to_visit.append(link)

        return scraped_contents

    def process_pdfs(self, pdf_paths: Optional[List[str]] = None) -> List[str]:
        """Process PDF files and save their content as text files."""
        processed_files = []

        # Use provided paths or all PDFs in the pdf_dir
        if pdf_paths is None:
            pdf_paths = [str(pdf) for pdf in self.pdf_dir.glob("*.pdf")]

        for pdf_path in pdf_paths:
            try:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()

                # Extract filename
                filename = os.path.basename(pdf_path)
                basename = os.path.splitext(filename)[0]
                output_path = self.raw_dir / f"{basename}.txt"

                # Combine all pages
                text_content = ""
                for doc in documents:
                    text_content += doc.page_content + "\n\n"

                # Save as text
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"Source: {pdf_path}\n\n{text_content}")

                processed_files.append(str(output_path))

            except Exception as e:
                logger.error(f"Error processing PDF {pdf_path}: {str(e)}")

        return processed_files

    def chunk_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Split documents into chunks for better retrieval."""
        chunks = []

        for file_path in file_paths:
            try:
                # Load the document
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract metadata
                metadata = self._extract_metadata(content, file_path)

                # Create document with metadata
                doc = Document(page_content=content, metadata=metadata)

                # Split into chunks
                chunked_docs = self.text_splitter.split_documents([doc])

                # Add to chunks list with metadata
                chunks.extend([{
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", file_path),
                    "chunk_id": f"{os.path.basename(file_path)}_{i}",
                    "title": doc.metadata.get("title", ""),
                    "topic": doc.metadata.get("topic", "general")
                } for i, doc in enumerate(chunked_docs)])

            except Exception as e:
                logger.error(f"Error chunking {file_path}: {str(e)}")

        # Save chunked documents
        for i, chunk in enumerate(chunks):
            with open(self.processed_dir / f"chunk_{i}.txt", 'w', encoding='utf-8') as f:
                f.write(
                    f"Source: {chunk['source']}\nTitle: {chunk['title']}\nTopic: {chunk['topic']}\n\n{chunk['content']}")

        return chunks

    def _get_filename_from_url(self, url: str) -> str:
        """Generate a safe filename from a URL."""
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        path = re.sub(r'[^\w\-_.]', '_', path)

        if not path:
            path = 'index'

        # Add a hash of the full URL to ensure uniqueness
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]

        return f"{path}_{url_hash}"

    def process_all(self) -> List[Dict[str, Any]]:
        """Process website, PDFs, and DOCX files, then chunk everything."""
        # Process website
        scraped_files = self.scrape_website()

        # Process PDFs and DOCX
        pdf_files = self.process_pdfs()
        docx_files = self.process_docx()

        # Combine all files and chunk them
        all_files = scraped_files + pdf_files + docx_files
        return self.chunk_documents(all_files)

    def process_docx(self, docx_paths: Optional[List[str]] = None) -> List[str]:
        """Process DOCX files and save their content as text files."""
        processed_files = []

        # Use provided paths or all DOCX in the pdf_dir (we're reusing the pdf_dir for all documents)
        if docx_paths is None:
            docx_paths = [str(docx) for docx in self.pdf_dir.glob("*.docx")]

        for docx_path in docx_paths:
            try:
                loader = Docx2txtLoader(docx_path)
                documents = loader.load()

                # Extract filename
                filename = os.path.basename(docx_path)
                basename = os.path.splitext(filename)[0]
                output_path = self.raw_dir / f"{basename}.txt"

                # Combine all content
                text_content = ""
                for doc in documents:
                    text_content += doc.page_content + "\n\n"

                # Save as text
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"Source: {docx_path}\n\n{text_content}")

                processed_files.append(str(output_path))

            except Exception as e:
                logger.error(f"Error processing DOCX {docx_path}: {str(e)}")

        return processed_files

    def _extract_metadata(self, content: str, file_path: str) -> Dict[str, str]:
        """Extract metadata from content."""
        # Default metadata
        metadata = {
            "source": file_path,
            "title": os.path.basename(file_path),
            "topic": "general"
        }

        # Try to extract title from content
        lines = content.strip().split('\n')
        if lines and len(lines[0]) < 100:
            metadata["title"] = lines[0].strip()

        # Topic detection
        content_lower = content.lower()
        topics = {
            "account opening": ["open account", "account opening", "new account"],
            "bank account": ["bank account", "add bank", "link bank"],
            "trading": ["trade", "trading", "buy", "sell", "order"],
            "funds": ["fund", "deposit", "withdraw", "money", "payment"],
            "kyc": ["kyc", "know your customer", "verification", "documents"],
            "fees": ["fee", "charge", "pricing", "cost", "brokerage"],
        }

        for topic, keywords in topics.items():
            if any(keyword in content_lower for keyword in keywords):
                metadata["topic"] = topic
                break

        return metadata