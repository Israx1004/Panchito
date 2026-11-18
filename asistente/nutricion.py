"""Calculadora nutrimental: calorías, carbohidratos, proteína, grasa, fibra y carga glucémica.

Sin LLM a propósito: las cifras salen de una tabla local (datos/alimentos.json, valores aproximados USDA)
y de reglas, para que el modelo no invente números. El LLM solo interpreta el resultado.

Entiende cantidades escritas como "200 g de arroz", "2 tortillas", "una taza de frijoles", "1/2 taza de avena",
"medio plátano". Si no hay cantidad, usa la porción estándar del alimento y lo marca.
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache

from config import DATOS

NUTRIENTES = ("kcal", "carbohidratos", "proteina", "grasa", "fibra")
PALABRAS_NUM = {"un": 1, "una": 1, "uno": 1, "medio": 0.5, "media": 0.5, "dos": 2, "tres": 3, "cuatro": 4,
                "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10}
# unidad escrita -> (campo de peso en alimentos.json, gramos si el alimento no lo define)
UNIDADES = {"taza": ("taza_g", 240), "vaso": ("vaso_g", 240), "pieza": ("pieza_g", None),
            "rebanada": ("rebanada_g", None), "cucharada": ("cucharada_g", 15), "cucharadita": ("cucharadita_g", 5),
            "lata": ("lata_g", 355), "punado": ("punado_g", 28), "porcion": ("porcion_g", None),
            "plato": ("plato_g", None), "orden": ("orden_g", None), "bolsa": ("bolsa_g", 45),
            "botella": ("botella_g", 355), "copa": ("copa_g", 150), "bola": ("bola_g", 66), "paquete": ("paquete_g", None)}
AVISO_ESTIMADO = ("Estas cifras son solo un estimado: los valores reales cambian según la receta, la marca, "
                  "la preparación y el tamaño de la porción.")
_NUM = r"\b(\d+(?:\.\d+)?|" + "|".join(PALABRAS_NUM) + r")"
_UNIDAD = (r"(kg|kilos?|g|gr|grs|gramos?|ml|mililitros?|tazas?|vasos?|piezas?|rebanadas?|cucharadas?"
           r"|cucharaditas?|latas?|punados?|porcion(?:es)?|platos?|orden(?:es)?|bolsas?|botellas?|copas?|bolas?|paquetes?)")
CANTIDAD = re.compile(_NUM + r"\s*" + _UNIDAD + r"?\s*(?:de\s+)?(?:(?:la|el|los|las|un|una)\s+)?$")
PIDE_CALCULO = re.compile(r"calori|kcal|carbohidrato|\bcarbos?\b|hidratos|proteina|fibra|grasa|macro|nutriente"
                          r"|cuant[oa]s?\b|calcula")


def texto_normal(texto: str) -> str:
    """Minúsculas, sin acentos; conserva decimales y convierte fracciones (1/2 -> 0.5)."""
    t = unicodedata.normalize("NFKD", (texto or "").casefold())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"(\d+)\s*/\s*(\d+)", lambda m: str(int(m.group(1)) / max(int(m.group(2)), 1)), t)
    t = re.sub(r"(\d),(\d)", r"\1.\2", t)
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)
    return " ".join(re.sub(r"[^a-z0-9.]+", " ", t).split())


@lru_cache(maxsize=1)
def tabla() -> list[dict]:
    return json.loads((DATOS / "alimentos.json").read_text(encoding="utf-8"))["alimentos"]


@lru_cache(maxsize=1)
def _patrones() -> list[tuple[re.Pattern, dict]]:
    pares = [(texto_normal(s), a) for a in tabla() for s in a["sinonimos"]]
    pares.sort(key=lambda p: -len(p[0]))  # el sinónimo más largo gana: "pan integral" antes que "pan"
    # Plural opcional en cada palabra: "tortillas de harina" coincide con "tortilla de harina".
    return [(re.compile(r"\b" + r"\s".join(re.escape(w) + r"(?:s|es)?" for w in s.split()) + r"\b"), a)
            for s, a in pares]


def _gramos(alimento: dict, numero: str | None, unidad: str | None) -> tuple[float, str, bool]:
    """Devuelve (gramos, cantidad legible, es_estimado)."""
    if numero is None:
        return alimento["porcion_g"], "porción estándar", True
    n = PALABRAS_NUM.get(numero) or float(numero)
    u = (unidad or "").rstrip("s") if unidad not in ("g", "gr", "grs", "kg") else unidad
    if u in ("g", "gr", "grs", "gramo"):
        return n, f"{n:g} g", False
    if u in ("kg", "kilo"):
        return n * 1000, f"{n:g} kg", False
    if u in ("ml", "mililitro"):
        return n, f"{n:g} ml", False
    u = {"porcione": "porcion", "ordene": "orden"}.get(u, u)
    if not u and "rebanada_g" in alimento and "pieza_g" in alimento:
        u = "rebanada"  # "una pizza" o "un pastel" sin unidad: se cuenta 1 rebanada y se muestra así
    campo, defecto = UNIDADES.get(u, ("pieza_g", None))
    peso = alimento.get(campo) or defecto or alimento.get("pieza_g") or alimento["porcion_g"]
    nombre = {"porcion": "porción", "punado": "puñado"}.get(u, u) or ("pieza" if "pieza_g" in alimento else "porción")
    plural = ("es" if nombre in ("porción", "orden") else "s") if n > 1 else ""
    return n * peso, f"{n:g} {nombre.replace('ó', 'o') if plural == 'es' else nombre}{plural}", False


def extraer_alimentos(mensaje: str) -> list[dict]:
    """Alimentos de la tabla mencionados en el mensaje, con su cantidad en gramos."""
    t = texto_normal(mensaje)
    encontrados, ocupado = [], []
    for patron, alimento in _patrones():
        for m in patron.finditer(t):
            if any(m.start() < fin and m.end() > ini for ini, fin in ocupado):
                continue
            ocupado.append((m.start(), m.end()))
            encontrados.append((m.start(), m.end(), alimento))
    encontrados.sort(key=lambda e: e[0])
    resultado, fin_previo = [], 0
    for ini, fin, alimento in encontrados:
        if resultado and t[fin_previo:ini].strip() == "de":
            fin_previo = fin  # "enchiladas de pollo", "caldo de res": describe al platillo anterior, no se suma
            continue
        cantidad = CANTIDAD.search(t[fin_previo:ini])
        gramos, legible, estimado = _gramos(alimento, *(cantidad.groups() if cantidad else (None, None)))
        resultado.append({"alimento": alimento, "gramos": round(gramos, 1), "cantidad": legible, "estimado": estimado})
        fin_previo = fin
    return resultado


def nivel_carga(cg: float) -> str:
    return "baja" if cg <= 10 else "media" if cg < 20 else "alta"


def calcular(items: list[dict]) -> dict:
    """Suma nutrientes por porción. Carga glucémica = IG x (carbohidratos - fibra) / 100."""
    filas, total, cg_total, con_ig = [], dict.fromkeys(NUTRIENTES, 0.0), 0.0, False
    for it in items:
        a, f = it["alimento"], it["gramos"] / 100
        fila = {k: round(a[k] * f, 1) for k in NUTRIENTES}
        fila["kcal"] = round(fila["kcal"])
        if a.get("ig") is not None:
            fila["carga_glucemica"] = round(a["ig"] * max(fila["carbohidratos"] - fila["fibra"], 0) / 100, 1)
            cg_total += fila["carga_glucemica"]
            con_ig = True
        filas.append(fila | {"nombre": a["nombre"], "cantidad": it["cantidad"], "gramos": it["gramos"],
                             "estimado": it["estimado"]})
        for k in NUTRIENTES:
            total[k] += fila[k]
    total = {k: round(v, 1) for k, v in total.items()}
    total["kcal"] = round(total["kcal"])
    # Platillos con carbohidratos pero sin índice glucémico: la carga glucémica total queda incompleta.
    sin_ig = [f["nombre"] for f in filas if "carga_glucemica" not in f and f["carbohidratos"] - f["fibra"] >= 5]
    return {"filas": filas, "total": total, "carga_glucemica": round(cg_total, 1) if con_ig else None,
            "sin_ig": sin_ig}


def _linea(d: dict) -> str:
    return (f"{d['kcal']} kcal | carbohidratos {d['carbohidratos']} g | proteína {d['proteina']} g | "
            f"grasa {d['grasa']} g | fibra {d['fibra']} g")


def _texto_carga(calc: dict) -> str:
    cg = calc["carga_glucemica"]
    if calc.get("sin_ig"):  # incompleta: no se califica como baja/media/alta
        con = [f["nombre"] for f in calc["filas"] if "carga_glucemica" in f]
        return (f"Carga glucémica parcial: {cg:g}, solo de {', '.join(con)}; sin dato de índice glucémico para "
                f"{', '.join(calc['sin_ig'])}, así que la carga real de la comida es mayor")
    return f"Carga glucémica estimada: {cg:g} ({nivel_carga(cg)})"


def formatear(calc: dict) -> str:
    """Tabla fija que se muestra al usuario tal cual (no pasa por el LLM)."""
    lineas = ["Cálculo nutrimental ESTIMADO (valores aproximados, no exactos):"]
    for f in calc["filas"]:
        marca = " *" if f["estimado"] else ""
        peso = "" if f["cantidad"].endswith((" g", " ml", " kg")) else f" ({f['gramos']:g} g)"
        lineas.append(f"- {f['nombre'].capitalize()}, {f['cantidad']}{peso}{marca}: {_linea(f)}")
    if len(calc["filas"]) > 1:
        lineas.append(f"Total: {_linea(calc['total'])}")
    if calc["carga_glucemica"] is not None:
        lineas.append(_texto_carga(calc))
    if any(f["estimado"] for f in calc["filas"]):
        lineas.append("* Sin cantidad indicada: se usó una porción estándar.")
    lineas.append(AVISO_ESTIMADO)
    return "\n".join(lineas)


def resumen(calc: dict) -> str:
    """Versión corta para el contexto del LLM."""
    nombres = ", ".join(f"{f['nombre']} {f['gramos']:g} g" for f in calc["filas"])
    cg = "; " + _texto_carga(calc).lower() if calc["carga_glucemica"] is not None else ""
    return f"[calculadora, valores ESTIMADOS] {nombres}. Total: {_linea(calc['total'])}{cg}"


def ficha(alimento: dict) -> str:
    """Datos de una porción estándar, como evidencia para preguntas generales sobre un alimento."""
    f = calcular([{"alimento": alimento, "gramos": alimento["porcion_g"], "cantidad": "porción", "estimado": True}])
    return f"[tabla, valores ESTIMADOS] {alimento['nombre']}, porción de {alimento['porcion_g']:g} g: {_linea(f['total'])}"
