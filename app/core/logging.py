import logging
import uuid
import sys
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from app.core.config import settings

# Configure logging format
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | [%(name)s] [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("custom_llm_robot")

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that assigns a unique X-Request-ID to each incoming request
    and logs request timing & execution.
    """
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        
        logger.info(f"Incoming Request [{request_id}]: {request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            logger.info(f"Response Sent [{request_id}]: Status {response.status_code}")
            return response
        except Exception as exc:
            logger.error(f"Request Failed [{request_id}]: {str(exc)}", exc_info=True)
            raise exc
