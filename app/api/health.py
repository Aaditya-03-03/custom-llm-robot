from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check endpoint to verify system status."""
    return {"status": "ok", "system": "Custom LLM Robot"}
