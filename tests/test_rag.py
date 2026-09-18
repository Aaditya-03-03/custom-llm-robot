import os
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings
from app.rag.loader import DocumentLoader
from app.rag.chunker import TextChunker
from app.rag.embeddings import EmbeddingManager
from app.rag.vector_store import VectorStoreManager
from app.rag.retriever import KnowledgeRetriever, FALLBACK_RESPONSE

client = TestClient(app)

def test_document_loader(tmp_path):
    # Test loading markdown file
    md_file = tmp_path / "test_doc.md"
    md_file.write_text("# Robot Info\nThe robot uses ESP32 controllers.", encoding="utf-8")
    
    doc_info = DocumentLoader.load_file(str(md_file))
    assert doc_info["filename"] == "test_doc.md"
    assert "ESP32 controllers" in doc_info["content"]
    assert len(doc_info["hash"]) == 64  # SHA256 string length

def test_text_chunker():
    doc_info = {
        "filename": "robot_test.md",
        "path": "/tmp/robot_test.md",
        "content": "Paragraph 1 about Unity interface.\n\nParagraph 2 about ESP32 controllers.",
        "hash": "dummyhash123"
    }
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_document(doc_info)
    
    assert len(chunks) > 0
    assert chunks[0]["metadata"]["document"] == "robot_test.md"
    assert "chunk_id" in chunks[0]

def test_embedding_manager():
    mgr = EmbeddingManager()
    vec = mgr.embed_text("Test robot query")
    assert isinstance(vec, list)
    assert len(vec) == 384  # BGE-small embedding dimension

def test_knowledge_query_success(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")
    
    # 1. Prepare temp document directory
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    doc_file = docs_dir / "robot_info.md"
    doc_file.write_text("The robot uses a Meta Quest headset for human-robot interaction.", encoding="utf-8")
    
    db_dir = tmp_path / "vector_db"
    
    # 2. Ingest document
    store_mgr = VectorStoreManager(db_dir=str(db_dir))
    store_mgr.sync_documents(documents_dir=str(docs_dir))
    
    # 3. Patch KnowledgeRetriever to use our temp store manager
    monkeypatch.setattr("app.api.knowledge.KnowledgeRetriever", lambda: KnowledgeRetriever(vector_store_manager=store_mgr))
    
    # 4. Execute query
    response = client.post(
        "/api/v1/knowledge/query",
        json={"question": "What platform is used for human-robot interaction?"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["sources"]) > 0
    assert data["sources"][0]["document"] == "robot_info.md"
    assert "score" in data["sources"][0]
    assert data["sources"][0]["score"] > 0.35

def test_knowledge_query_out_of_scope(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")
    
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    doc_file = docs_dir / "robot_info.md"
    doc_file.write_text("The robot uses a Meta Quest headset.", encoding="utf-8")
    
    db_dir = tmp_path / "vector_db"
    store_mgr = VectorStoreManager(db_dir=str(db_dir))
    store_mgr.sync_documents(documents_dir=str(docs_dir))
    
    monkeypatch.setattr("app.api.knowledge.KnowledgeRetriever", lambda: KnowledgeRetriever(vector_store_manager=store_mgr))
    
    # Query completely unrelated info
    response = client.post(
        "/api/v1/knowledge/query",
        json={"question": "What is the secret recipe for authentic Italian pizza dough?"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == FALLBACK_RESPONSE
    assert len(data["sources"]) == 0
