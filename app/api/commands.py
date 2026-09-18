from fastapi import APIRouter

router = APIRouter()

@router.post("/commands")
async def execute_command():
    """Direct robot command execution endpoint."""
    return {"status": "command_received"}
