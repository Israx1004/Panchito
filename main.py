
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests  # noqa: E402

from config import EMBED_MODEL, EMOJI, KEEP_ALIVE, MODELO, OLLAMA_URL  # noqa: E402

NODOS_QUE_RESPONDEN = {"agente_educacion", "agente_nutricion", "agente_calculo", "agente_medicacion", "agente_monitoreo", "agente_emergencia"}


def preparar() -> None:
    """Carga modelos e índice antes de la primera pregunta (la carga inicial tarda ~20 s una sola vez)."""
    from asistente import indice
    from asistente.herramientas import iniciar_bd

    try:
        requests.post(f"{OLLAMA_URL}/api/generate", json={"model": MODELO, "keep_alive": KEEP_ALIVE}, timeout=300)
        requests.post(f"{OLLAMA_URL}/api/embed", json={"model": EMBED_MODEL, "input": "hola", "keep_alive": KEEP_ALIVE},
                      timeout=300)
    except requests.ConnectionError:
        sys.exit("No hay conexión con Ollama. Ábrelo y vuelve a intentar.")
    iniciar_bd()
    indice.cargar()
    indice.vectorizar("nutrición")


def responder(app, mensaje: str, historial: list[tuple[str, str]], sesion: str = "local") -> None:
    from asistente.agentes import NOMBRE_AGENTE

    t0 = time.perf_counter()
    print("Asistente: ", end="", flush=True)
    estado, escrito = {}, False
    for modo, dato in app.stream({"mensaje": mensaje, "historial": historial, "sesion": sesion},
                                 stream_mode=["custom", "messages", "values"]):
        if modo == "custom":                       # texto fijo: aviso de emergencia, fuera de tema o tabla nutrimental
            print(dato["texto"], flush=True)
        elif modo == "messages":
            chunk, meta = dato
            if meta.get("langgraph_node") in NODOS_QUE_RESPONDEN and chunk.content:
                print(EMOJI.sub("", chunk.content).replace("**", "").replace("#", ""), end="", flush=True)
                escrito = True
        else:
            estado = dato
    if escrito:
        print()
    agente = next((n for n in reversed(estado.get("ruta", [])) if n in NOMBRE_AGENTE), "")
    print(f"[agente: {NOMBRE_AGENTE.get(agente, '-')} · {time.perf_counter() - t0:.1f} s]")
    historial.append((mensaje, estado.get("respuesta", "")))


def main() -> None:
    print("Asistente de nutrición y diabetes · cargando modelo...", flush=True)
    preparar()
    from asistente.grafo import app

    print(f"Listo ({MODELO}). Pregunta sobre alimentación, calorías y nutrientes, o diabetes. Escribe 'salir' para terminar.")
    historial: list[tuple[str, str]] = []
    sesion = time.strftime("chat-%Y%m%d-%H%M%S")  # perfil y mediciones de glucosa son de esta conversación
    while True:
        try:
            mensaje = input("\nTú: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if mensaje.lower() in ("salir", "exit", "quit"):
            break
        if mensaje:
            responder(app, mensaje, historial, sesion)


if __name__ == "__main__":
    main()
