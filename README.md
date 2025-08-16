# Tu Casa – NL Search API

API para:
1) Recibir una consulta en lenguaje natural (`q`)
2) (Luego) generar un **SearchPlan** con OpenAI
3) Mapear a **query params** y construir una **URL** de resultados

## Dev local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
