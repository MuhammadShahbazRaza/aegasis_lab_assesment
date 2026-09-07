from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.observability.context import current_run_id
from app.observability.logging import get_logger

log = get_logger(__name__)


class ApiError(Exception):
    status_code = status.HTTP_400_BAD_REQUEST

    def __init__(self, message: str, detail: object = None):
        super().__init__(message)
        self.message = message
        self.detail = detail


class NotFound(ApiError):
    status_code = status.HTTP_404_NOT_FOUND


class Conflict(ApiError):
    status_code = status.HTTP_409_CONFLICT


class UpstreamUnavailable(ApiError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


def register_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.message, "detail": exc.detail},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": "request validation failed",
                "detail": [
                    {
                        "field": ".".join(str(p) for p in err["loc"][1:]) or "<body>",
                        "problem": err["msg"],
                    }
                    for err in exc.errors()
                ],
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        # Unmatched routes and 405s are raised by Starlette itself, so without this
        # they would come back in FastAPI's default {"detail": ...} shape while every
        # other error uses {"error": ..., "detail": ...}.
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": str(exc.detail), "detail": None},
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception(
            "unhandled error", extra={"path": request.url.path, "error": str(exc)}
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal server error",
                "detail": {"run_id": current_run_id()},
            },
        )
