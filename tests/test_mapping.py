from app.schemas import PropertyFilters
from app.mapping import build_search_url
from app.settings import settings

def test_build_url_basic(monkeypatch):
    monkeypatch.setenv("BASE_URL", "https://tucasapy.com")
    monkeypatch.setenv("RESULTS_PATH", "/buscar")

    # recargar settings con los envs mockeados
    from importlib import reload
    import app.settings as s
    reload(s)

    f = PropertyFilters(
        operation="venta",
        property_type=["departamento", "apto"],
        city=["Asunción"],
        min_price=50000,
        max_price=120000,
        min_bedrooms=2,
        amenities=["piscina","seguridad 24/7","cochera"]
    )
    url = build_search_url(f)
    assert url.startswith("https://tucasapy.com/buscar?")
    assert "operacion=venta" in url
    assert "tipo=departamento" in url
    assert "ciudad=asuncion" in url
    assert "min_precio=50000" in url
