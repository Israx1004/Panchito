"""Técnicas con LLM (Tareas 3-5): Zero-shot, Tagging y Clustering."""
from __future__ import annotations

import re
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans

from config import embeddings, llm, norm

CATEGORIAS = ["Síntomas", "Dieta", "Medicación", "Emergencia", "Seguimiento"]

# Señales de alarma: se revisan sin LLM para que un error del modelo no oculte una emergencia.
SENALES_ALARMA = [
    "inconsciente", "perdio el conocimiento", "desmay", "no puede tragar", "no puedo tragar",
    "convulsion", "respiro rapido", "respiracion rapida", "confundid", "dolor en el pecho",
    "dificultad para hablar", "aliento a fruta",
]

_llm_det = None


def _det():
    global _llm_det
    if _llm_det is None:
        _llm_det = llm(temperature=0, num_predict=150)
    return _llm_det


def hay_senal_alarma(mensaje: str) -> bool:
    m = norm(mensaje)
    return any(norm(s) in m for s in SENALES_ALARMA)


DESCRIPCIONES = {
    "Síntomas": "malestar o síntomas que siente el paciente",
    "Dieta": "alimentos, comidas, bebidas, porciones",
    "Medicación": "insulina, dosis, pastillas, medicamentos, efectos secundarios",
    "Emergencia": "inconsciencia, confusión grave, no poder tragar, vómito con respiración rápida",
    "Seguimiento": "mediciones y registro de glucosa, consultas, tendencias",
}


def zero_shot(mensaje: str, categorias: list[str] = CATEGORIAS) -> str:
    opciones = "\n".join(f"- {c}: {DESCRIPCIONES[c]}" if c in DESCRIPCIONES else f"- {c}" for c in categorias)
    prompt = ("Clasifica el mensaje en UNA de estas categorías:\n"
              f"{opciones}\nResponde solo con el nombre de la categoría.\n"
              f"Mensaje: {mensaje}\nCategoría:")
    salida = norm(_det().invoke(prompt).content)
    return next((c for c in categorias if norm(c) in salida), categorias[-1])


class DatosPaciente(BaseModel):
    nivel_glucosa: str | None = Field(description="valor de glucosa con unidades, ej. '210 mg/dl'; null si no se menciona")
    sintomas: list[str] = Field(description="síntomas mencionados; lista vacía si no hay")
    gravedad: Literal["leve", "moderada", "grave"] = Field(description="gravedad estimada del mensaje")
    tipo_diabetes: str | None = Field(description="'tipo 1', 'tipo 2' o null si no se menciona")


def tagging(mensaje: str) -> DatosPaciente:
    try:
        d = _det().with_structured_output(DatosPaciente).invoke(
            f"Extrae los datos clínicos del mensaje del paciente:\n{mensaje}")
    except Exception:  # el modelo puede devolver JSON inválido
        return DatosPaciente(nivel_glucosa=None, sintomas=[], gravedad="leve", tipo_diabetes=None)
    # Validación posterior: el modelo a veces rellena campos con texto ("no especificado").
    if d.nivel_glucosa and not re.search(r"\d{2,3}", d.nivel_glucosa):  # "tipo 2" no es un valor de glucosa
        d.nivel_glucosa = None
    tipo = re.search(r"tipo\s*([12])", norm(d.tipo_diabetes or ""))
    d.tipo_diabetes = f"tipo {tipo.group(1)}" if tipo else None
    d.sintomas = [s for s in d.sintomas if "especific" not in norm(s) and "diabetes" not in norm(s)]
    return d


def clustering(textos: list[str], n_grupos: int = 2) -> dict[int, list[str]]:
    # Prefijo de tarea recomendado por EmbeddingGemma; vectores normalizados (distancia coseno).
    vectores = np.array(embeddings().embed_documents(["task: clustering | query: " + t for t in textos]))
    vectores /= np.linalg.norm(vectores, axis=1, keepdims=True)
    grupos = KMeans(n_clusters=n_grupos, random_state=42, n_init=10).fit_predict(vectores)
    return {int(g): [t for t, k in zip(textos, grupos) if k == g] for g in sorted(set(grupos))}


# ---------- Chat: Zero-shot + Tagging en UNA llamada corta (Tarea 5, optimizado por latencia) ----------
# En este CPU cada token del prompt cuesta ~10 ms y cada token generado ~60 ms; por eso el prompt es mínimo
# y la salida estructurada usa un formato compacto de 4 campos separados por ";" en lugar de JSON.
TEMAS_CHAT = ["educacion", "sintomas", "dieta", "medicacion", "emergencia", "seguimiento", "fuera"]
NOMBRE_TEMA = {"educacion": "Educación", "sintomas": "Síntomas", "dieta": "Dieta", "medicacion": "Medicación",
               "emergencia": "Emergencia", "seguimiento": "Seguimiento", "fuera": "Fuera de tema"}


class Clasificacion(BaseModel):
    """Categoría (zero-shot) + datos clínicos (tagging)."""
    categoria: str
    glucosa: int | None = None
    gravedad: Literal["leve", "moderada", "grave"] = "leve"
    sintomas: list[str] = []
    tipo: int | None = None


PROMPT_CLASIFICAR = """Asistente de nutrición y diabetes. Clasifica el mensaje del usuario. Categorías:
dieta: alimentos, comidas, bebidas, calorías, carbohidratos, proteína, fibra, porciones, peso, plan de alimentación
educacion: preguntas generales sobre diabetes o salud (qué es, tipos, causas, complicaciones, hemoglobina glucosilada)
sintomas: el usuario reporta malestar que siente ahora | seguimiento: reporta sus mediciones de glucosa, registro, tendencia
medicacion: insulina, pastillas, medicamentos | emergencia: inconsciencia, desmayo, no poder tragar
fuera: no trata de alimentación, nutrición, ejercicio, salud ni diabetes
Responde una línea: categoria;glucosa mg/dl o 0;leve|moderada|grave;síntomas separados por coma o -
Ejemplo: sintomas;250;moderada;sed, mareo"""

_llm_clasif = None


def _clasificador():
    global _llm_clasif
    if _llm_clasif is None:
        _llm_clasif = llm(temperature=0, num_predict=24)
    return _llm_clasif


SIN_DATO = re.compile(r"^(-|no aplica|ninguno|ninguna|n/?a|no se recomienda|sin sintomas|no)$")
TEMA_SALUD = re.compile(r"diabet|glucosa|azucar|insulin|medic|pastilla|sintoma|dolor|mare|sed|cansad|hambre|comer|comida"
                        r"|desayun|cenar|almorz|aliment|dieta|bebida|fruta|ejercicio|peso|salud|presion|herida"
                        r"|calori|kcal|carbohidrato|carbos|proteina|fibra|grasa|vitamin|nutri|macro|porcion|verdura")
COMIDA = re.compile(r"hambre|comer|comida|desayun|cenar|almorz|aliment|dieta|bebida|fruta|verdura"
                    r"|calori|kcal|carbohidrato|carbos|proteina|fibra|grasa|vitamin|nutri|macro|porcion|peso")


def clasificar(mensaje: str, previo: str = "") -> Clasificacion:
    """Zero-shot (categoría sin ejemplos de entrenamiento) + Tagging (extracción estructurada), una sola llamada."""
    texto = f"{previo} / {mensaje}" if previo else mensaje
    salida = _clasificador().invoke([("system", PROMPT_CLASIFICAR), ("human", texto)]).content
    partes = [p.strip() for p in salida.strip().splitlines()[0].split(";")] if salida.strip() else []
    partes += [""] * (4 - len(partes))
    cat = next((t for t in TEMAS_CHAT if t in norm(partes[0])), "educacion")
    grav = next((g for g in ("grave", "moderada", "leve") if g in norm(partes[2])), "leve")
    sintomas = [s.strip() for s in partes[3].split(",") if s.strip() and not SIN_DATO.match(norm(s))][:4]
    c = Clasificacion(categoria=cat, gravedad=grav, sintomas=sintomas)
    # Validación posterior con el texto del usuario: el texto manda sobre lo que "imagine" el modelo.
    n = norm(mensaje)
    if cat == "fuera" and TEMA_SALUD.search(n):  # no rechazar una pregunta que sí menciona el dominio
        c.categoria = "dieta" if COMIDA.search(n) else "educacion"
    # Pregunta general ("¿cuáles son los síntomas de...?") sin hablar de sí mismo: es educación, no un reporte.
    general = re.match(r"(cuales|cual|que|por que|como se|cuando) ", n)
    propio = re.search(r"\b(me|mi|mis|tengo|siento|estoy|tuve)\b", n)
    if c.categoria in ("sintomas", "seguimiento") and general and not propio:
        c.categoria = "educacion"
    m = re.search(r"\b(\d{2,3})\b", n)
    if m and 20 <= int(m.group(1)) <= 700 and re.search(r"glucosa|azucar|mg|dl|glucometro|nivel", n):
        c.glucosa = int(m.group(1))
    if t := re.search(r"tipo\s*([12])", n):
        c.tipo = int(t.group(1))
    return c
