import os
import hashlib
import logging
from typing import List, Dict, Any

logger = logging.getLogger("custom_llm_robot.rag.loader")

class DocumentLoader:
    """
    Multi-format document extractor supporting .md, .txt, .pdf, and .docx files.
    Calculates SHA256 hashes for incremental indexing.
    """

    @staticmethod
    def calculate_file_hash(filepath: str) -> str:
        """Calculate SHA256 hash of a file on disk."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    @classmethod
    def load_file(cls, filepath: str) -> Dict[str, Any]:
        """Load and extract text from a single file based on extension."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Document file not found: {filepath}")

        ext = os.path.splitext(filepath)[1].lower()
        filename = os.path.basename(filepath)
        content = ""

        try:
            if ext in [".md", ".txt"]:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            elif ext == ".pdf":
                try:
                    import pypdf
                    reader = pypdf.PdfReader(filepath)
                    pages_text = [page.extract_text() or "" for page in reader.pages]
                    content = "\n".join(pages_text)
                except ImportError:
                    logger.error("pypdf package is required for PDF parsing.")
                    raise
            elif ext == ".docx":
                try:
                    import docx
                    doc = docx.Document(filepath)
                    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                    content = "\n".join(paragraphs)
                except ImportError:
                    logger.error("python-docx package is required for DOCX parsing.")
                    raise
            else:
                logger.warning(f"Unsupported file extension '{ext}' for file {filename}")
                return {}

            content = content.strip()
            file_hash = cls.calculate_file_hash(filepath)

            return {
                "filename": filename,
                "path": filepath,
                "content": content,
                "hash": file_hash
            }
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {e}")
            raise

    @classmethod
    def load_directory(cls, dir_path: str) -> List[Dict[str, Any]]:
        """Load all supported documents from a directory recursively."""
        documents = []
        if not os.path.exists(dir_path):
            logger.warning(f"Documents directory '{dir_path}' does not exist.")
            return documents

        supported_extensions = {".md", ".txt", ".pdf", ".docx"}

        for root, _, files in os.walk(dir_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in supported_extensions:
                    full_path = os.path.join(root, file)
                    doc_info = cls.load_file(full_path)
                    if doc_info and doc_info.get("content"):
                        documents.append(doc_info)

        logger.info(f"Loaded {len(documents)} document(s) from '{dir_path}'")
        return documents
