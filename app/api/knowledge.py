import logging
from fastapi import APIRouter, HTTPException, status
from app.schemas.knowledge import KnowledgeQueryRequest, KnowledgeQueryResponse, DocumentSource
from app.rag.retriever import KnowledgeRetriever
from app.llm.ollama_provider import OllamaProviderError

logger = logging.getLogger("custom_llm_robot.api.knowledge")
router = APIRouter(prefix="/api/v1", tags=["Knowledge"])

@router.post("/knowledge/query", response_model=KnowledgeQueryResponse, summary="Query project knowledge base")
async def query_knowledge_base(request: KnowledgeQueryRequest):
    """
    Query project documentation via RAG system. Returns answer generated strictly from
    relevant vector database context along with source document metadata.
    """
    logger.info(f"Received Knowledge Query: '{request.question}'")
    
    try:
        retriever = KnowledgeRetriever()
        answer, sources = await retriever.query(request.question)
        
        doc_sources = [
            DocumentSource(
                document=s["document"],
                chunk_id=s.get("chunk_id"),
                score=s.get("score")
            )
            for s in sources
        ]
        
        logger.info(f"Knowledge Query completed. Returned {len(doc_sources)} source document reference(s).")
        return KnowledgeQueryResponse(answer=answer, sources=doc_sources)

    except OllamaProviderError as e:
        logger.error(f"LLM Provider Error during RAG query: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Local LLM service unavailable: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error executing knowledge query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while querying the knowledge base."
        )
