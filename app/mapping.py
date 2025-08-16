import logging
from typing import Dict, Any
from .schemas import PropertyFilters, SearchPlan
from .utils import slugify, qs
from .filters_catalog import (
    canonize, canonize_list, PROPERTY_TYPES_CANON, AMENITIES_CANON, OPERATIONS_CANON
)
from .settings import settings
from .errors import MappingError

logger = logging.getLogger(__name__)

def filters_to_params(f: PropertyFilters) -> Dict[str, Any]:
    try:
        operation = canonize(f.operation, OPERATIONS_CANON) if isinstance(f.operation, str) else f.operation
        property_type = canonize_list(f.property_type, PROPERTY_TYPES_CANON)
        amenities = canonize_list(f.amenities, AMENITIES_CANON)

        params: Dict[str, Any] = {
            "operacion": operation,
            "tipo": [slugify(t) for t in property_type] or None,
            "ciudad": [slugify(c) for c in f.city] or None,
            "barrio": [slugify(b) for b in f.neighborhood] or None,
            "min_precio": f.min_price,
            "max_precio": f.max_price,
            "min_dorms": f.min_bedrooms,
            "min_banos": f.min_bathrooms,
            "min_cocheras": f.min_parking,
            "min_m2": f.min_m2,
            "max_m2": f.max_m2,
            "amenities": [slugify(a) for a in amenities] or None,
            "ordenar_por": f.sort_by,
        }
        logger.debug("filters_to_params -> %s", params)
        return params
    except Exception as e:
        raise MappingError(f"Error al mapear filtros a parámetros: {e}") from e

def build_search_url(filters: PropertyFilters) -> str:
    base = (settings.BASE_URL or "").rstrip("/")
    path = settings.RESULTS_PATH if settings.RESULTS_PATH.startswith("/") else f"/{settings.RESULTS_PATH}"
    params = filters_to_params(filters)
    query = qs(params)
    url = f"{base}{path}?{query}" if base else (f"{path}?{query}" if query else path)
    logger.info("URL construida: %s", url)
    return url

def soft_prefs_to_sort(soft_prefs) -> str | None:
    if getattr(soft_prefs, "luxury", None) == "high":
        return "lujo"
    if getattr(soft_prefs, "affordability", None) == "cheap":
        return "economia"
    if getattr(soft_prefs, "recency", None) == "newest":
        return "reciente"
    return None

def plan_to_url(plan: SearchPlan) -> str:
    f = plan.must_filters
    if not f.sort_by:
        f.sort_by = soft_prefs_to_sort(plan.soft_prefs)
    return build_search_url(f)
