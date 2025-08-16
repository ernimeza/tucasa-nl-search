import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from .errors import AppError

logger = logging.getLogger(__name__)

def init_error_handlers(app):
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError):
        logger.error("%s(code=%s): %s", exc.__class__.__name__, exc.code, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.code, "message": exc.message},
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        logger.error("ValidationError: %s", exc.errors())
        return JSONResponse(
            status_code=422,
            content={"error": "VALIDATION_ERROR", "message": "Payload inválido", "details": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception):
        logger.exception("Unexpected error")
        return JSONResponse(
            status_code=500,
            content={"error": "INTERNAL_ERROR", "message": "Ocurrió un error inesperado"},
        )
