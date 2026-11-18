"""Herramientas (Tarea 5): RAG, búsqueda, base de datos y calculadora clínica.

El RAG integra dos fuentes:
  - guías educativas (datos/guias/*.md)
  - preguntas y respuestas curadas en las Tareas 6-7 (resultados/teacher_responses_v2.jsonl),
    que ya pasaron por deduplicación, filtrado y corrección de seguridad.
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime

from asistente import indice
from config import DATOS, RESULTADOS

DB_PATH = RESULTADOS / "pacientes.db"


def rag(consulta, guia: str | tuple[str, ...] | None = None, k: int = 3) -> list[dict]:
    """RAG sobre fragmentos aprobados (índice semántico); acepta texto o el vector ya calculado de la pregunta.
    `guia` filtra por tema: nutricion, macronutrientes, dieta, insulina, emergencias, seguimiento."""
    vector = indice.vectorizar(consulta) if isinstance(consulta, str) else consulta
    return indice.buscar(vector, guia=guia, k=k)


def buscar_servicios(tipo: str = "urgencias") -> list[dict]:
    """Directorio local de servicios médicos (sustituible por una API web)."""
    servicios = json.loads((DATOS / "servicios.json").read_text(encoding="utf-8"))
    return [s for s in servicios if s["tipo"] == tipo] or servicios


def buscar_info_medicamento(nombre: str) -> str | None:
    fichas = json.loads((DATOS / "medicamentos.json").read_text(encoding="utf-8"))
    return next((t for clave, t in fichas.items() if clave in (nombre or "").lower()), None)


def iniciar_bd() -> None:
    """Crea la BD del paciente real. No hay datos de ejemplo: el perfil (tipo de diabetes, peso, estatura) y las
    mediciones de glucosa salen solo de lo que el usuario dice, y se guardan por sesión de chat."""
    if DB_PATH.exists():
        con = sqlite3.connect(DB_PATH)
        columnas = [c[1] for c in con.execute("PRAGMA table_info(glucosa)")]
        con.close()
        if columnas and "sesion" not in columnas:  # BD antigua con paciente demo ficticio: se respalda y se rehace
            DB_PATH.rename(DB_PATH.with_name(f"pacientes_demo_respaldo_{datetime.now():%Y%m%d_%H%M%S}.db"))
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS perfil(sesion TEXT PRIMARY KEY, tipo TEXT, peso_kg REAL, estatura_m REAL,
                                          actualizado TEXT);
        CREATE TABLE IF NOT EXISTS glucosa(id INTEGER PRIMARY KEY AUTOINCREMENT, sesion TEXT,
                                           fecha TEXT, mg_dl REAL, momento TEXT);
    """)
    con.commit()
    con.close()


def obtener_perfil(sesion: str) -> dict:
    """Perfil que el usuario dio en esta sesión; campos ausentes = no los ha dicho."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    fila = con.execute("SELECT tipo, peso_kg, estatura_m FROM perfil WHERE sesion=?", (sesion,)).fetchone()
    con.close()
    return {k: v for k, v in dict(fila).items() if v is not None} if fila else {}


def actualizar_perfil(sesion: str, **campos) -> None:
    campos = {k: v for k, v in campos.items() if v is not None and k in ("tipo", "peso_kg", "estatura_m")}
    if not campos:
        return
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT OR IGNORE INTO perfil(sesion) VALUES (?)", (sesion,))
    for k, v in campos.items():
        con.execute(f"UPDATE perfil SET {k}=?, actualizado=? WHERE sesion=?",
                    (v, datetime.now().isoformat(timespec="minutes"), sesion))
    con.commit()
    con.close()


def extraer_perfil(mensaje: str) -> dict:
    """Tipo de diabetes, peso y estatura escritos por el usuario ("tengo tipo 1", "peso 60 kg", "mido 1.65")."""
    from asistente.nutricion import texto_normal

    t = texto_normal(mensaje)
    perfil = {}
    if m := re.search(r"\btipo\s*(1|2|uno|dos)\b", t):
        perfil["tipo"] = {"uno": "1", "dos": "2"}.get(m.group(1), m.group(1))
    elif re.search(r"\bgestacional\b", t):
        perfil["tipo"] = "gestacional"
    m = (re.search(r"\b(\d{2,3}(?:\.\d+)?)\s*(?:kg|kilos?|kilogramos?)\b(?!\s+de\b)", t)
         or re.search(r"\bpeso\s+(\d{2,3}(?:\.\d+)?)\b", t))
    if m and 30 <= float(m.group(1)) <= 250:
        perfil["peso_kg"] = float(m.group(1))
    m = (re.search(r"(?:mido|estatura|altura)\D{0,12}?([12]\.\d{1,2})\b", t)
         or re.search(r"\b([12]\.\d{2})\s*(?:m|mts?|metros?)\b", t))
    cm = re.search(r"(?:mido|estatura|altura)\D{0,12}?\b(1\d{2})\s*(?:cm|centimetros)?\b", t)
    if m and 1.0 <= float(m.group(1)) <= 2.3:
        perfil["estatura_m"] = float(m.group(1))
    elif cm:
        perfil["estatura_m"] = int(cm.group(1)) / 100
    return perfil


def historial_glucosa(sesion: str, n: int = 7) -> list[dict]:
    """Mediciones que el usuario reportó en esta sesión (monitoreo en tiempo real del paciente)."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    filas = con.execute("SELECT fecha, mg_dl, momento FROM glucosa WHERE sesion=? ORDER BY id DESC LIMIT ?",
                        (sesion, n)).fetchall()
    con.close()
    return [dict(f) for f in reversed(filas)]


def registrar_glucosa(mg_dl: float, sesion: str, momento: str = "reportado en chat") -> None:
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT INTO glucosa(sesion,fecha,mg_dl,momento) VALUES (?,?,?,?)",
                (sesion, datetime.now().isoformat(timespec="minutes"), mg_dl, momento))
    con.commit()
    con.close()


def calcular_imc(peso_kg: float, estatura_m: float) -> float:
    return round(peso_kg / estatura_m ** 2, 1)


def carga_glucemica(indice_glucemico: float, carbohidratos_g: float) -> float:
    """CG = IG × carbohidratos (g) / 100. Baja <=10, media 11-19, alta >=20."""
    return round(indice_glucemico * carbohidratos_g / 100, 1)
