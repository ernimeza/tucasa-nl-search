from typing import Optional

class AppError(Exception):
    status_code: int = 400
    code: str = "APP_ERROR"

    def __init__(self, message: str, *, code: Optional[str] = None, status_code: Optional[int] = None):
        super().__init__(message)
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code
        self.message = message

class MappingError(AppError):
    code = "MAPPING_ERROR"
    status_code = 400

class PlannerError(AppError):
    code = "PLANNER_ERROR"
    status_code = 422

class ExternalServiceError(AppError):
    code = "EXTERNAL_SERVICE_ERROR"
    status_code = 502
