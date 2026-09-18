"""
Main entry point for the Custom LLM Robot application API.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path if not present
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from fastapi import FastAPI
from app.api import chat, commands, health

app = FastAPI(
    title="Custom LLM Robot API",
    description="LLM-powered autonomous robotic controller API",
    version="0.1.0",
)

app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(commands.router, prefix="/api", tags=["Commands"])

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Welcome to Custom LLM Robot API",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
