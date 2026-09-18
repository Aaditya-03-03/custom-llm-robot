import re
import logging
from typing import List, Dict, Any
from app.rag.config import rag_settings

logger = logging.getLogger("custom_llm_robot.rag.chunker")

class TextChunker:
    """
    Splits long document text into overlapping chunks while preserving metadata.
    """

    def __init__(
        self,
        chunk_size: int = rag_settings.RAG_CHUNK_SIZE,
        chunk_overlap: int = rag_settings.RAG_CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a document dictionary into structured text chunks.
        """
        text = doc_info.get("content", "").strip()
        filename = doc_info.get("filename", "document")
        filepath = doc_info.get("path", "")
        doc_hash = doc_info.get("hash", "")

        if not text:
            return []

        # Split into paragraph blocks or sentences
        paragraphs = re.split(r'\n\s*\n', text)
        chunks_text = []
        current_chunk = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If a single paragraph exceeds chunk_size, split by sentences or hard break
            if len(para) > self.chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sentence in sentences:
                    if current_len + len(sentence) > self.chunk_size and current_chunk:
                        chunks_text.append("\n".join(current_chunk))
                        # Keep overlap from end of previous chunk
                        overlap_str = current_chunk[-1] if current_chunk else ""
                        current_chunk = [overlap_str] if len(overlap_str) <= self.chunk_overlap else []
                        current_len = sum(len(s) for s in current_chunk)

                    current_chunk.append(sentence)
                    current_len += len(sentence)
            else:
                if current_len + len(para) > self.chunk_size and current_chunk:
                    chunks_text.append("\n".join(current_chunk))
                    overlap_str = current_chunk[-1] if current_chunk else ""
                    current_chunk = [overlap_str] if len(overlap_str) <= self.chunk_overlap else []
                    current_len = sum(len(s) for s in current_chunk)

                current_chunk.append(para)
                current_len += len(para)

        if current_chunk:
            chunks_text.append("\n".join(current_chunk))

        # Format chunk objects with metadata
        result_chunks = []
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', filename)
        
        for idx, chunk_str in enumerate(chunks_text):
            chunk_id = f"{clean_name}_chunk_{idx}"
            result_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_str,
                "metadata": {
                    "document": filename,
                    "path": filepath,
                    "document_hash": doc_hash,
                    "chunk_id": chunk_id,
                    "chunk_index": idx
                }
            })

        logger.debug(f"Document '{filename}' split into {len(result_chunks)} chunk(s).")
        return result_chunks

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk multiple document dictionaries."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
