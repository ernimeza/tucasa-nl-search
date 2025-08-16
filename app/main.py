import os, json, time, logging, re, unicodedata
from typing import Optional, List, Literal, Tuple, Dict, Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field, conint, confloat, ValidationError

# ========== ENV ==========
def _env_list(key: str, default: str = "") -> List[str]:
    raw = os.getenv(key, default)
    if not raw:
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
BASE_URL: str = (os.getenv("BASE_URL") or "").rstrip("/")
RESULTS_PATH: str = os.getenv("RESULTS_PATH", "")  # NO usamos aquí (porque tu portal usa path por segmentos)
CORS_ALLOW_ORIGINS = _env_list("CORS_ALLOW_ORIGINS", "*")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# ========== LOG + REQ-ID ==========
class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id", str(uuid4()))
        request.state.request_id = rid
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled exception")
            raise
        dur_ms = int((time.perf_counter() - start) * 1000)
        response.headers["X-Request-ID"] = rid
        logger.info("%s %s -> %s (%d ms) | req=%s",
                    request.method, request.url.path, response.status_code, dur_ms, rid)
        return response

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
)
logger = logging.getLogger("tucasa-nl")

# ========== ERRORES ==========
class AppError(Exception):
    status_code: int = 400
    code: str = "APP_ERROR"
    def __init__(self, message: str, *, code: Optional[str] = None, status_code: Optional[int] = None):
        super().__init__(message)
        if code: self.code = code
        if status_code: self.status_code = status_code
        self.message = message

class MappingError(AppError):
    code = "MAPPING_ERROR"; status_code = 400

class PlannerError(AppError):
    code = "PLANNER_ERROR"; status_code = 422

class ExternalServiceError(AppError):
    code = "EXTERNAL_SERVICE_ERROR"; status_code = 502

# ========== MODELOS ==========
Operation = Literal["venta", "alquiler"]
Currency = Literal["usd", "gs"]

class PropertyFilters(BaseModel):
    # Principales
    operation: Operation | None = None
    property_type: List[str] = Field(default_factory=list)   # "departamentos", "casas", ...
    city: List[str] = Field(default_factory=list)            # slugs: asuncion, luque, ...
    neighborhood: List[str] = Field(default_factory=list)    # slugs: villa-morra, ... (solo si asuncion)

    # Precio
    currency: Optional[Currency] = None                     # usd|gs -> "$"/"GS"
    min_price: Optional[confloat(ge=0)] = None
    max_price: Optional[confloat(ge=0)] = None

    # Dormitorios
    bedrooms_token: Optional[str] = None                    # "monoambiente" | "1".."10" | "+10"
    # opcionalmente, el planner podría poner rangos:
    min_bedrooms: Optional[conint(ge=0)] = None
    max_bedrooms: Optional[conint(ge=0)] = None

    # Metros / Hectáreas
    min_m2: Optional[conint(ge=0)] = None
    max_m2: Optional[conint(ge=0)] = None
    min_hectares: Optional[confloat(ge=0)] = None
    max_hectares: Optional[confloat(ge=0)] = None

    # Otros
    furnished: Optional[Literal["si","no"]] = None          # AM: Sí/No
    amenities: List[str] = Field(default_factory=list)      # A: solo 1 (tomamos la primera)
    styles: List[str] = Field(default_factory=list)         # E: solo 1 (primera)
    condition: List[str] = Field(default_factory=list)      # ES: solo 1 (primera)
    min_floor: Optional[conint(ge=0)] = None                # PIMI
    max_floor: Optional[conint(ge=0)] = None                # PIMA

    # Orden (no participa en la URL de tu portal actual)
    sort_by: Optional[str] = None

class SoftPrefs(BaseModel):
    luxury: Optional[Literal["low","medium","high"]] = None
    affordability: Optional[Literal["cheap","fair","expensive"]] = None
    safety: Optional[Literal["low","medium","high"]] = None
    recency: Optional[Literal["newest","any"]] = None
    price_per_m2_quantile: Optional[Literal["bottom20","top30","top10"]] = None

class SearchPlan(BaseModel):
    intent: Optional[Operation] = None
    locale: str = "es-PY"
    must_filters: PropertyFilters = Field(default_factory=PropertyFilters)
    soft_prefs: SoftPrefs = Field(default_factory=SoftPrefs)
    explain_to_user: Optional[str] = None

class NLQueryRequest(BaseModel):
    q: str = Field(description="Consulta del usuario en lenguaje natural")

class NLPlanResponse(BaseModel):
    url: str
    filters: PropertyFilters
    explain_to_user: Optional[str] = None
    debug_plan: Optional[SearchPlan] = None

# ========== CATÁLOGOS/SINÓNIMOS ==========
PROPERTY_TYPES_CANON = {
    "casas": ["casa", "casas"],
    "departamentos": ["departamento", "departamentos", "apto", "apto.", "depto", "depto."],
    "duplex": ["duplex", "dúplex"],
    "terrenos": ["terreno", "terrenos", "lote", "lotes"],
    "oficinas": ["oficina", "oficinas"],
    "locales": ["local", "locales", "local comercial", "locales comerciales", "comercial"],
    "edificios": ["edificio", "edificios"],
    "paseos": ["paseo", "paseos", "paseo comercial", "shopping", "mall"],
    "depositos": ["deposito", "depósito", "depósitos", "depósito"],
    "quintas": ["quinta", "quintas"],
    "estancias": ["estancia", "estancias"],
}

# canonical slug -> etiqueta EXACTA para tu portal (param "A")
AMENITY_LABEL = {
    "acceso-controlado": "Acceso controlado",
    "area-de-coworking": "Área de coworking",
    "area-de-parrilla": "Área de parrilla",
    "area-de-yoga": "Área de yoga",
    "area-verde": "Área verde",
    "bar": "Bar",
    "bodega": "Bodega",
    "cancha-de-padel": "Cancha de pádel",
    "cancha-de-tenis": "Cancha de tenis",
    "cancha-de-futbol": "Cancha de fútbol",
    "cerradura-digital": "Cerradura digital",
    "cine": "Cine",
    "club-house": "Club house",
    "estacionamiento-techado": "Estacionamiento techado",
    "generador": "Generador",
    "gimnasio": "Gimnasio",
    "laguna-artificial": "Laguna artificial",
    "laguna-natural": "Laguna natural",
    "lavanderia": "Lavandería",
    "parque-infantil": "Parque infantil",
    "piscina": "piscina",
    "quincho": "Quincho",
    "salon-de-eventos": "Salón de eventos",
    "sala-de-juegos": "Sala de juegos",
    "sala-de-masajes": "Sala de masajes",
    "sala-de-reuniones": "Sala de reuniones",
    "sauna": "Sauna",
    "seguridad-24-7": "Seguridad 24/7",
    "solarium": "Solarium",
    "spa": "Spa",
    "terraza": "Terraza",
    "wifi": "Wi-Fi",
    "cafe": "Café",
    "business-center": "Business center",
}

AMENITIES_CANON = {k: [k.replace("-", " "), AMENITY_LABEL[k].lower()] for k in AMENITY_LABEL.keys()}

STYLE_LABEL = {
    "moderna": "Moderna",
    "minimalista": "Minimalista",
    "clasica": "Clásica",
    "de-campo": "De campo",
    "de-playa": "De playa",
    "de-lujo": "De lujo",
    "de-verano": "De verano",
    "para-inversion": "Para inversión",
    "sustentable": "Sustentable",
    "prefabricada": "Prefabricada",
    "inteligente": "inteligente",
}
STYLE_CANON = {k: [k.replace("-", " "), STYLE_LABEL[k].lower()] for k in STYLE_LABEL.keys()}

CONDITION_LABEL = {
    "a-estrenar": "A estrenar",
    "perfecto": "Perfecto",
    "muy-bueno": "Muy bueno",
    "bueno": "Bueno",
    "desarrollo": "Desarrollo",
    "remodelada": "Remodelada",
    "regular": "Regular",
    "le-falta-trabajo": "Le falta trabajo",
    "en-construccion": "En construcción",
}
CONDITION_CANON = {k: [k.replace("-", " "), CONDITION_LABEL[k].lower()] for k in CONDITION_LABEL.keys()}

CURRENCY_CANON = { "usd": ["usd", "$", "dolares", "dólares"], "gs": ["gs", "pyg", "guaranies", "guaraníes", "₲"] }
FURNISHED_CANON = { "si": ["si","sí","amoblado","amueblado"], "no": ["no","sin muebles","no amoblado","no amueblado"] }

# ========== UTILS ==========
def slugify(value: str) -> str:
    if not value: return ""
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return value

def canonize(value: Optional[str], table: Dict[str, List[str]]) -> Optional[str]:
    if not value: return None
    v = value.strip().lower()
    for canon, syns in table.items():
        if v == canon or v in syns: return canon
    return None

def canonize_list(values: Optional[List[str]], table: Dict[str, List[str]]) -> List[str]:
    out: List[str] = []
    for v in values or []:
        c = canonize(v, table)
        if c and c not in out: out.append(c)
    return out

def qs(params: Dict[str, Any]) -> str:
    from urllib.parse import urlencode
    clean = {}
    for k, v in params.items():
        if v is None: continue
        if isinstance(v, list) and len(v) == 0: continue
        clean[k] = v
    return urlencode(clean, doseq=True)

def normalize_bedrooms_token(token: Optional[str]) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    if not token: return None, None, None
    t = token.strip().lower()
    if t in {"monoambiente", "studio", "estudio"}: return "monoambiente", 0, 0
    if t in {"+10", "10+"}: return "+10", 10, None
    if t.isdigit():
        n = int(t); n = max(0, n)
        return str(n), n, n
    return t, None, None

# ========== URL BUILDER ESPECÍFICO DE TU PORTAL ==========
def build_portal_url(filters: PropertyFilters) -> str:
    """
    Construye: https://tucasapy.com/{oper}/{tipo1}/{ciudad1}/{barrio1?}?...
      - PD2..PD5: tipos extra
      - C2..C5: ciudades extra
      - B, B2..B5: barrios (solo si ciudad1 == 'asuncion')
      - H, H2..H5: habitaciones (tokens)
      - divisa: "$" | "GS"
      - A, AM, E, ES, PL, PIMI, PIMA, M2MI, M2MA, HMI, HMA
    """

    # 1) Canonizar algunos campos a nuestros canónicos
    types_canon = canonize_list(filters.property_type, PROPERTY_TYPES_CANON)
    ams_canon   = canonize_list(filters.amenities, AMENITIES_CANON)
    styles_canon= canonize_list(filters.styles, STYLE_CANON)
    conds_canon = canonize_list(filters.condition, CONDITION_CANON)
    currency_c  = canonize(filters.currency, CURRENCY_CANON) if isinstance(filters.currency, str) else filters.currency
    furnished_c = canonize(filters.furnished, FURNISHED_CANON) if isinstance(filters.furnished, str) else filters.furnished

    # 2) Path segments
    oper = (filters.operation or "venta").strip().lower()  # por defecto "venta"
    tipo1 = types_canon[0] if types_canon else ""          # ej: "departamentos"
    city1 = (filters.city[0].strip().lower() if filters.city else "")
    barrio1 = (filters.neighborhood[0].strip().lower() if filters.neighborhood else "")

    # El barrio va en el path SOLO si la ciudad principal es asuncion y hay barrio
    include_barrio_in_path = (city1 == "asuncion" and bool(barrio1))

    # construimos el path
    path_parts = [p for p in [oper, tipo1, city1, (barrio1 if include_barrio_in_path else None)] if p]
    if not BASE_URL:
        raise MappingError("BASE_URL no configurada. Define BASE_URL en variables de entorno.")
    base_path = "/".join([BASE_URL.rstrip("/")] + path_parts)

    # 3) Query params según tu esquema
    params: Dict[str, Any] = {}

    # Tipos extra: PD2..PD5 (máx 4 adicionales)
    for i, t in enumerate(types_canon[1:5], start=2):
        params[f"PD{i}"] = t

    # Ciudades extra: C2..C5
    for i, c in enumerate([slugify(x) for x in filters.city[1:5]], start=2):
        params[f"C{i}"] = c

    # Barrios: si ciudad principal es asuncion
    if city1 == "asuncion" and filters.neighborhood:
        # B (principal) y B2..B5 (resto)
        params["B"] = slugify(filters.neighborhood[0])
        for i, b in enumerate([slugify(x) for x in filters.neighborhood[1:5]], start=2):
            params[f"B{i}"] = b

    # Habitaciones: H, H2..H5
    tokens: List[str] = []
    tok, tmin, tmax = normalize_bedrooms_token(filters.bedrooms_token) if filters.bedrooms_token else (None, None, None)
    if tok: tokens.append(tok)
    # si vinieron min/max y forman exacto, añadimos (evita duplicados)
    if filters.min_bedrooms is not None and filters.max_bedrooms is not None and filters.min_bedrooms == filters.max_bedrooms:
        t2 = str(filters.min_bedrooms)
        if t2 not in tokens: tokens.append(t2)
    # cargar hasta 5 (H, H2..H5)
    if tokens:
        params["H"] = tokens[0]
        for i, t in enumerate(tokens[1:5], start=2):
            params[f"H{i}"] = t

    # divisa: "$" | "GS"
    if currency_c == "usd":
        params["divisa"] = "$"
    elif currency_c == "gs":
        params["divisa"] = "GS"

    # Precios
    if filters.min_price is not None: params["Precio-min"] = filters.min_price
    if filters.max_price is not None: params["Precio-max"] = filters.max_price

    # Amenidad preferida (una): A
    if ams_canon:
        first_amenity = ams_canon[0]   # canonical slug
        params["A"] = AMENITY_LABEL.get(first_amenity, first_amenity)

    # Amoblado: AM (Sí|No)
    if furnished_c == "si": params["AM"] = "Sí"
    elif furnished_c == "no": params["AM"] = "No"

    # Estilo (uno): E
    if styles_canon:
        params["E"] = STYLE_LABEL.get(styles_canon[0], styles_canon[0])

    # Estado (uno): ES
    if conds_canon:
        params["ES"] = CONDITION_LABEL.get(conds_canon[0], conds_canon[0])

    # Plantas (PL) -> opcional (no lo estamos mapeando aún)
    # if filters.floors is not None: params["PL"] = filters.floors

    # Piso mínimo/máximo
    if filters.min_floor is not None: params["PIMI"] = filters.min_floor
    if filters.max_floor is not None: params["PIMA"] = filters.max_floor

    # m2
    if filters.min_m2 is not None: params["M2MI"] = filters.min_m2
    if filters.max_m2 is not None: params["M2MA"] = filters.max_m2

    # hectáreas
    if filters.min_hectares is not None: params["HMI"] = filters.min_hectares
    if filters.max_hectares is not None: params["HMA"] = filters.max_hectares

    # Nota: NO seteamos "página", lo maneja tu front.

    query = qs(params)
    return f"{base_path}?{query}" if query else base_path

# ========== PLANNER (OpenAI) ==========
PLANNER_SYS = """Eres un planificador de búsqueda inmobiliaria para Paraguay.
Devuelves SOLO un JSON SearchPlan, sin texto adicional.
Usa:
- operation: 'venta' o 'alquiler'
- property_type: de la lista proporcionada (plural, ej. 'departamentos', 'casas', etc.)
- city y neighborhood como slugs en minúsculas (ej: 'asuncion', 'villa-morra')
- currency: 'usd' o 'gs' si el usuario menciona $ o GS/guaraníes.
- bedrooms_token: 'monoambiente' | '1'..'10' | '+10' si aplica.
- amenities/styles/condition: usa las claves canónicas entregadas (no etiquetas).
- Si no es claro un campo, déjalo vacío (null o lista vacía).
Responde ÚNICAMENTE JSON válido.
"""

def build_planner_user(q: str) -> str:
    catalogs = {
        "property_types": list(PROPERTY_TYPES_CANON.keys()),
        "amenities": list(AMENITIES_CANON.keys()),
        "styles": list(STYLE_LABEL.keys()),
        "conditions": list(CONDITION_LABEL.keys()),
        "operation": ["venta", "alquiler"],
        "currency": ["usd", "gs"],
        "city_hint": "Usa slugs en minúsculas sin acentos (ej: san-lorenzo, villa-elisa)."
    }
    template = {
        "task": "Mapear consulta a SearchPlan para el portal Tu Casa (Paraguay).",
        "query": q,
        "catalogs": catalogs,
        "output_schema_hint": "Sigue el esquema SearchPlan (must_filters, soft_prefs, etc.)."
    }
    return json.dumps(template, ensure_ascii=False)

def run_planner(q: str) -> SearchPlan:
    if not OPENAI_API_KEY:
        raise PlannerError("Falta OPENAI_API_KEY en el entorno para usar el planner.")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": PLANNER_SYS},
                {"role": "user", "content": build_planner_user(q)},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        plan_dict = json.loads(raw)
        plan = SearchPlan.model_validate(plan_dict)
        return plan
    except ValidationError as ve:
        logger.error("Planner JSON inválido: %s", ve)
        raise PlannerError("Planner devolvió JSON inválido.")
    except Exception:
        logger.exception("Error llamando a OpenAI")
        raise ExternalServiceError("Error al consultar OpenAI")

# ========== FASTAPI ==========
app = FastAPI(title="Tu Casa - NL Search API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if CORS_ALLOW_ORIGINS == ["*"] else CORS_ALLOW_ORIGINS,
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
app.add_middleware(RequestIdMiddleware)

@app.exception_handler(AppError)
async def handle_app_error(request: Request, exc: AppError):
    rid = getattr(request.state, "request_id", "-")
    logger.error("%s(code=%s): %s | req=%s", exc.__class__.__name__, exc.code, exc.message, rid)
    return JSONResponse(status_code=exc.status_code, content={"error": exc.code, "message": exc.message})

@app.exception_handler(Exception)
async def handle_unexpected(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", "-")
    logger.exception("Unexpected error | req=%s", rid)
    return JSONResponse(status_code=500, content={"error": "INTERNAL_ERROR", "message": "Ocurrió un error inesperado"})

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/nl-search/plan", response_model=NLPlanResponse)
def nl_search_plan(req: NLQueryRequest):
    # 1) Planner
    plan = run_planner(req.q)

    # 2) Construir URL exacta del portal
    url = build_portal_url(plan.must_filters)

    # 3) Responder (puedes quitar debug_plan en prod)
    return NLPlanResponse(url=url, filters=plan.must_filters, explain_to_user=plan.explain_to_user, debug_plan=plan)
