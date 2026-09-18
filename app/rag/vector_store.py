import os
import logging
import chromadb
from typing import List, Dict, Any, Tuple
from app.rag.config import rag_settings
from app.rag.loader import DocumentLoader
from app.rag.chunker import TextChunker
from app.rag.embeddings import EmbeddingManager

logger = logging.getLogger("custom_llm_robot.rag.vector_store")

COLLECTION_NAME = "robot_knowledge_base"

class VectorStoreManager:
    """
    Direct ChromaDB vector store manager handling persistent storage and incremental hashed syncing.
    """

    def __init__(
        self,
        db_dir: str = rag_settings.VECTOR_DB_DIR,
        embedding_manager: EmbeddingManager = None
    ):
        self.db_dir = db_dir
        os.makedirs(self.db_dir, exist_ok=True)
        
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self._client = chromadb.PersistentClient(path=self.db_dir)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    @property
    def collection(self):
        return self._collection

    def get_indexed_document_hashes(self) -> Dict[str, str]:
        """
        Query ChromaDB metadata to build a map of {filename: document_hash}.
        """
        try:
            results = self.collection.get(include=["metadatas"])
            metadatas = results.get("metadatas", [])
            doc_hashes = {}
            for meta in metadatas:
                if meta and "document" in meta and "document_hash" in meta:
                    doc_hashes[meta["document"]] = meta["document_hash"]
            return doc_hashes
        except Exception as e:
            logger.error(f"Error reading indexed document hashes: {e}")
            return {}

    def sync_documents(self, documents_dir: str = rag_settings.DOCUMENTS_DIR) -> Dict[str, int]:
        """
        Perform incremental update:
        - Load disk documents.
        - Compare SHA256 hashes with existing ChromaDB contents.
        - Add new, update modified, and delete removed document chunks.
        """
        disk_documents = DocumentLoader.load_directory(documents_dir)
        disk_doc_map = {doc["filename"]: doc for doc in disk_documents}
        indexed_hash_map = self.get_indexed_document_hashes()

        added_count = 0
        updated_count = 0
        deleted_count = 0

        # 1. Handle deleted files
        for filename in list(indexed_hash_map.keys()):
            if filename not in disk_doc_map:
                logger.info(f"Removing deleted document '{filename}' from ChromaDB.")
                self.collection.delete(where={"document": filename})
                deleted_count += 1

        # 2. Handle new and modified files
        chunker = TextChunker()
        chunks_to_insert = []

        for filename, doc_info in disk_doc_map.items():
            current_hash = doc_info["hash"]
            existing_hash = indexed_hash_map.get(filename)

            if existing_hash == current_hash:
                logger.debug(f"Document '{filename}' is unchanged. Skipping.")
                continue

            if existing_hash is not None:
                logger.info(f"Document '{filename}' modified. Updating chunks in ChromaDB.")
                self.collection.delete(where={"document": filename})
                updated_count += 1
            else:
                logger.info(f"Document '{filename}' is new. Indexing chunks into ChromaDB.")
                added_count += 1

            chunks = chunker.chunk_document(doc_info)
            chunks_to_insert.extend(chunks)

        # 3. Embed & insert new/updated chunks
        if chunks_to_insert:
            texts = [c["text"] for c in chunks_to_insert]
            ids = [c["chunk_id"] for c in chunks_to_insert]
            metadatas = [c["metadata"] for c in chunks_to_insert]

            embeddings = self.embedding_manager.embed_batch(texts)

            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.info(f"Successfully upserted {len(chunks_to_insert)} chunk(s) into ChromaDB.")

        return {
            "added": added_count,
            "updated": updated_count,
            "deleted": deleted_count,
            "total_chunks_inserted": len(chunks_to_insert)
        }

    def query(self, query_text: str, top_k: int = rag_settings.RAG_TOP_K) -> List[Dict[str, Any]]:
        """
        Query vector store using cosine similarity.
        Returns list of matching chunks with similarity score (0.0 to 1.0).
        """
        query_embedding = self.embedding_manager.embed_text(query_text)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        matches = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return matches

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i in range(len(ids)):
            # Convert cosine distance to cosine similarity (1.0 - distance)
            distance = distances[i]
            similarity_score = max(0.0, 1.0 - distance)
            
            matches.append({
                "chunk_id": ids[i],
                "text": documents[i],
                "metadata": metadatas[i],
                "score": round(float(similarity_score), 4)
            })

        return matches
