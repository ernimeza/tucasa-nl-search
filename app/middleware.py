import time
import logging
from uuid import uuid4
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from .logging_config import request_id_ctx

logger = logging.getLogger(__name__)

class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id", str(uuid4()))
        token = request_id_ctx.set(rid)
        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
        except Exception:
            logger.exception("Unhandled exception")
            raise
        finally:
            request_id_ctx.reset(token)
        dur_ms = int((time.perf_counter() - start) * 1000)
        response.headers["X-Request-ID"] = rid
        logger.info("%s %s -> %s (%d ms)", request.method, request.url.path, response.status_code, dur_ms)
        return response
