"""Índice semántico del asistente: RAG + Clustering con un solo embedding por pregunta.

Se construye una vez y se guarda en resultados/indice.json (se reconstruye solo si cambian las fuentes):
  - Documentos del RAG: guías educativas (datos/guias/*.md) y preguntas/respuestas curadas
    de las Tareas 6-7 (resultados/teacher_responses_v2.jsonl).
  - Clustering (T3/T5): K-Means sobre las preguntas curadas. Cada grupo se etiqueta con su tema
    mayoritario; en el chat, la pregunta se asigna al grupo más cercano y ese tema guía la búsqueda.

En el chat, la pregunta se convierte en vector UNA vez (~0.1 s) y ese vector sirve para el clustering y el RAG.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache

import numpy as np

from config import DATOS, RESULTADOS, embeddings, escribir_json, leer_jsonl

INDICE = RESULTADOS / "indice.json"
QA_CURADO = RESULTADOS / "teacher_responses_v2.jsonl"
N_GRUPOS = 6
# Tema de guía asociado a cada categoría de los datos curados.
TEMA_POR_CATEGORIA = {"dieta": "dieta", "medicacion": "insulina", "emergencia": "emergencias",
                      "seguimiento": "seguimiento", "sintomas": "seguimiento", "educacion": "seguimiento"}


def _fuentes() -> list:
    return sorted((DATOS / "guias").glob("*.md")) + ([QA_CURADO] if QA_CURADO.exists() else [])


def _documentos() -> list[dict]:
    docs = []
    for archivo in sorted((DATOS / "guias").glob("*.md")):
        for bloque in archivo.read_text(encoding="utf-8").split("\n\n"):
            if bloque.strip() and not bloque.startswith("#"):
                docs.append({"texto": " ".join(bloque.split()), "guia": archivo.stem, "fuente": "guia"})
    if QA_CURADO.exists():
        for r in leer_jsonl(QA_CURADO):
            docs.append({"texto": f"P: {r['pregunta']} R: {r['respuesta']}", "pregunta": r["pregunta"],
                         "categoria": r["categoria"], "guia": TEMA_POR_CATEGORIA.get(r["categoria"], "seguimiento"),
                         "fuente": "qa_curado"})
    return docs


def construir() -> dict:
    from sklearn.cluster import KMeans

    docs = _documentos()
    vec = np.array(embeddings().embed_documents([d["texto"] for d in docs]))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    qa = [i for i, d in enumerate(docs) if d["fuente"] == "qa_curado"]
    km = KMeans(n_clusters=N_GRUPOS, random_state=42, n_init=10).fit(vec[qa])
    grupos = []
    for g in range(N_GRUPOS):
        miembros = [docs[i] for i, k in zip(qa, km.labels_) if k == g]
        tema = Counter(m["categoria"] for m in miembros).most_common(1)[0][0]
        grupos.append({"grupo": g, "tema": tema, "tamano": len(miembros),
                       "ejemplos": [m["pregunta"] for m in miembros[:3]]})
    centros = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    data = {"fuentes": {str(p.name): p.stat().st_mtime for p in _fuentes()}, "docs": docs,
            "vectores": vec.round(6).tolist(), "centros": centros.round(6).tolist(), "grupos": grupos}
    escribir_json(INDICE, data)
    return data


@lru_cache(maxsize=1)
def cargar() -> dict:
    import json

    data = json.loads(INDICE.read_text(encoding="utf-8")) if INDICE.exists() else None
    actuales = {str(p.name): p.stat().st_mtime for p in _fuentes()}
    if not data or data["fuentes"] != actuales:
        data = construir()
    data["vectores"] = np.array(data["vectores"])
    data["centros"] = np.array(data["centros"])
    return data


@lru_cache(maxsize=1)
def _cliente():
    return embeddings()


def vectorizar(texto: str) -> np.ndarray:
    v = np.array(_cliente().embed_query(texto))
    return v / np.linalg.norm(v)


def grupo(vector: np.ndarray) -> dict:
    """Clustering: grupo de preguntas frecuentes más cercano a la pregunta."""
    idx = cargar()
    return idx["grupos"][int(np.argmax(idx["centros"] @ vector))]


def buscar(vector: np.ndarray, guia: str | tuple[str, ...] | None = None, k: int = 2) -> list[dict]:
    """RAG: los k fragmentos aprobados más similares; `guia` filtra por tema (uno o varios)."""
    idx = cargar()
    sims = idx["vectores"] @ vector
    guias = (guia,) if isinstance(guia, str) else guia
    orden = [i for i in np.argsort(-sims) if guias is None or idx["docs"][i]["guia"] in guias]
    return [idx["docs"][i] | {"similitud": float(sims[i])} for i in orden[:k]]


def similitud_maxima(vector: np.ndarray) -> float:
    return float((cargar()["vectores"] @ vector).max())
