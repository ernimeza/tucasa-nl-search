import re
import unicodedata
from urllib.parse import urlencode

def slugify(value: str) -> str:
    if not value:
        return ""
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return value

def qs(params: dict) -> str:
    clean = {}
    for k, v in params.items():
        if v is None:
            continue
        if isinstance(v, list) and len(v) == 0:
            continue
        clean[k] = v
    return urlencode(clean, doseq=True)
