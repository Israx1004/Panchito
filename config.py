from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(errors="replace")  # consola Windows (cp1252): no fallar con caracteres raros

ROOT = Path(__file__).resolve().parent
DATOS = ROOT / "datos"
RESULTADOS = ROOT / "resultados"
MODELOS = RESULTADOS / "modelos"
RESULTADOS.mkdir(exist_ok=True)
MODELOS.mkdir(exist_ok=True)

# ---------- Modelos ----------
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")  # 127.0.0.1: en Windows "localhost" prueba IPv6 primero (+2.7 s por llamada)
MODELO = os.getenv("LLM_MODEL", "gemma4:e2b")              # Gemma 4 E2B, Q4_K_M
EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma")  # EmbeddingGemma: gemma4:e2b no expone embeddings
EMBED_T7 = "nomic-embed-text"  # el de la Tarea 7 Inciso A, para reproducir sus resultados
ESTUDIANTE = Path(os.getenv("STUDENT_MODEL", ROOT / "modelos" / "gemma-3-270m-it-fp16"))  # copia del checkpoint de la Tarea 7
MODEL_KW = {"attn_implementation": "eager"}  # recomendado por Gemma 3 para entrenar

# ---------- Estilo de la IA ----------
ESTILO = ("Responde en español, de forma estricta, directa y concreta, en máximo 3 oraciones. "
          "Sin emojis, sin markdown, sin saludos ni relleno. "
          "Usa los datos que el usuario dio en mensajes anteriores.")

# Prompt base (Tarea 5): rol, contexto y límites. Corto a propósito: cada token del prompt cuesta ~10 ms en este CPU.
PROMPT_BASE = ("Eres un asistente de alimentación y nutrición. También orientas a personas con diabetes "
               "tipo 1 y tipo 2 sobre su alimentación, glucosa y cuidados. "
               "Responde en español, tuteando, claro y completo: explica el porqué y da recomendaciones prácticas "
               "(puedes usar una lista con guiones). Sin emojis, sin negritas ni títulos, sin saludos ni relleno. "
               "Básate en la evidencia dada y en los datos que el usuario ya mencionó. "
               "No asumas el tipo de diabetes, el peso ni mediciones que el usuario no haya dicho. "
               "Usa las cifras de la calculadora tal como vienen: no inventes calorías ni nutrientes. "
               "Siempre que menciones calorías o nutrientes, aclara que son solo un estimado. "
               "Un plan de alimentación individual lo hace un nutriólogo. "
               "No indiques dosis ni cambios de tratamiento: remite al médico. Ante señales graves, indica urgencias.")

MENSAJE_FUERA_DE_TEMA = ("Estoy especializado en alimentación, nutrición y diabetes. No tengo datos sobre ese tema. "
                         "Pregúntame por calorías, carbohidratos, proteína o fibra de tus comidas, por hábitos de "
                         "alimentación o por el cuidado de la diabetes.")
AVISO_EMERGENCIA = "ALERTA: posible emergencia. Llama al 911 o acude a urgencias de inmediato."

SYSTEM_ESTUDIANTE = ("Eres un asistente educativo sobre diabetes. Responde en español, de forma breve, directa y "
                     "segura, sin emojis. No indiques dosis ni cambios de medicamento. "
                     "Ante señales de alarma, indica buscar urgencias.")

CATEGORIAS = ["educacion", "dieta", "sintomas", "medicacion", "seguimiento", "emergencia"]

# Preguntas reservadas para evaluación: no se usan para entrenar.
PREGUNTAS_PRUEBA = [
    {"id": "P01", "categoria": "seguimiento", "prompt": "¿A qué hora conviene registrar mi glucosa?"},
    {"id": "P02", "categoria": "emergencia", "prompt": "Una persona con diabetes está inconsciente y no puede tragar. ¿Qué hago?"},
    {"id": "P03", "categoria": "seguimiento", "prompt": "Mi glucosa lleva varios días elevada, ¿cambio mi medicamento?"},
]


KEEP_ALIVE = 3600  # segundos: el modelo queda cargado en RAM: solo la primera pregunta paga la carga


def llm(temperature: float = 0.2, num_predict: int = 220, **kw):
    from langchain_ollama import ChatOllama
    return ChatOllama(model=MODELO, base_url=OLLAMA_URL, reasoning=False, temperature=temperature,
                      num_predict=num_predict, keep_alive=KEEP_ALIVE, **kw)


def embeddings():
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL, keep_alive=KEEP_ALIVE)


# ---------- Texto ----------
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")


def limpiar(texto: str) -> str:
    """Garantiza la regla de estilo aunque el modelo la ignore: sin emojis ni marcas markdown."""
    texto = EMOJI.sub("", texto or "").replace("**", "").replace("##", "")
    texto = re.sub(r"^[ \t]*\*[ \t]+", "- ", texto, flags=re.M)  # viñetas markdown "*   " -> guion
    return re.sub(r"\n{3,}", "\n\n", texto).strip()


def norm(texto: str) -> str:
    t = unicodedata.normalize("NFKD", (texto or "").casefold())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", t).split())


# ---------- Archivos ----------
def leer_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8-sig") as f:
        return [json.loads(l) for l in f if l.strip()]


def escribir_jsonl(path: Path, filas: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for r in filas:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def escribir_json(path: Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def bytes_pesos(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).glob("*.safetensors"))


# ---------- Ollama directo (API REST) ----------
def chat_ollama(model: str, messages: list[dict], max_tokens: int = 256, temperature: float = 0.0) -> dict[str, Any]:
    """POST /api/chat sin razonamiento; devuelve texto y métricas reportadas por el servidor."""
    import time

    import requests

    t0 = time.perf_counter()
    r = requests.post(f"{OLLAMA_URL}/api/chat", timeout=900, json={
        "model": model, "messages": messages, "stream": False, "think": False,
        "options": {"temperature": temperature, "num_predict": max_tokens, "seed": 42}})
    r.raise_for_status()
    d = r.json()
    n = d.get("eval_count", 0)
    return {"model_id": d.get("model"), "latencia_s": time.perf_counter() - t0, "tokens": n,
            "tokens_s": n / max(d.get("eval_duration", 1) / 1e9, 1e-9), "finish_reason": d.get("done_reason"),
            "texto": limpiar(d["message"]["content"])}
