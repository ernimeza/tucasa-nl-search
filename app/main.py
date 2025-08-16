from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="Tu Casa - NL Search API", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

# El endpoint real lo completaremos luego:
# - recibirá {"q": "texto del usuario"}
# - llamará a OpenAI para generar un SearchPlan
# - mapeará filtros -> URL con mapping.build_search_url(...)
@app.post("/nl-search/plan")
def nl_search_plan_stub():
    return JSONResponse(
        status_code=501,
        content={"message": "Not implemented yet. We'll add the LLM + mapping next."}
    )
