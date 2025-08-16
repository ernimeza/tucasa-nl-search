from pydantic import BaseModel, Field, conint, confloat
from typing import Literal, Optional

Operation = Literal["venta", "alquiler"]

class PropertyFilters(BaseModel):
    operation: Operation | None = None
    property_type: list[str] = Field(default_factory=list)
    city: list[str] = Field(default_factory=list)
    neighborhood: list[str] = Field(default_factory=list)

    min_price: Optional[confloat(ge=0)] = None
    max_price: Optional[confloat(ge=0)] = None
    min_bedrooms: Optional[conint(ge=0)] = None
    min_bathrooms: Optional[conint(ge=0)] = None
    min_parking: Optional[conint(ge=0)] = None
    min_m2: Optional[conint(ge=0)] = None
    max_m2: Optional[conint(ge=0)] = None

    amenities: list[str] = Field(default_factory=list)
    sort_by: Optional[str] = None          # relevancia|reciente|precio|lujo|economia

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
    debug_plan: Optional[SearchPlan] = None   # opcional para inspección
