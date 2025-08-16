# Catálogo de valores soportados y sinónimos comunes.

PROPERTY_TYPES_CANON = {
    "departamento": ["departamento", "apartamento", "depto", "apto"],
    "casa": ["casa"],
    "duplex": ["duplex", "dúplex"],
    "terreno": ["terreno", "lote", "lotes"],
    "oficina": ["oficina", "office"],
    "local-comercial": ["local", "local comercial", "comercial"],
}

AMENITIES_CANON = {
    "piscina": ["piscina", "pileta", "alberca"],
    "gimnasio": ["gimnasio", "gym"],
    "seguridad_24_7": ["seguridad 24/7", "guardia", "control de acceso", "porteria", "portería"],
    "cochera": ["cochera", "estacionamiento", "garage", "garaje", "parqueo"],
    "balcon": ["balcon", "balcón"],
    "ascensor": ["ascensor", "elevador"],
    "parrilla": ["parrilla", "quincho", "asador"],
    "patio": ["patio"],
    "jardin": ["jardin", "jardín"],
}

OPERATIONS_CANON = {
    "venta": ["venta", "comprar", "compra"],
    "alquiler": ["alquiler", "alquilar", "renta", "arrendar", "arriendo"],
}

def canonize(value: str | None, table: dict[str, list[str]]) -> str | None:
    if not value:
        return None
    v = value.strip().lower()
    for canon, syns in table.items():
        if v == canon or v in syns:
            return canon
    return None

def canonize_list(values: list[str] | None, table: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    for v in values or []:
        c = canonize(v, table)
        if c and c not in out:
            out.append(c)
    return out
