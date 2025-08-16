import logging
from contextvars import ContextVar

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.request_id = request_id_ctx.get()
        except Exception:
            record.request_id = "-"
        return True

def setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, (level or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | req=%(request_id)s | %(message)s",
    )
    for h in logging.getLogger().handlers:
        h.addFilter(RequestIdFilter())
