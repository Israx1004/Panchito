from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import requests  # noqa: E402

import main as cli  # noqa: E402  (reutiliza preparar() y los nodos que responden)
from config import EMOJI, MODELO, OLLAMA_URL  # noqa: E402

WEB = ROOT / "interfaz"
SESIONES: dict[str, list[tuple[str, str]]] = {}   # sesión -> historial (usuario, asistente)
TURNO = threading.Lock()   # el modelo corre en CPU: una respuesta a la vez; las demás esperan su turno
_preparado = threading.Event()
_app = None


def preparar() -> bool:
    """Carga modelo, embeddings, BD e índice una sola vez. Devuelve False si Ollama no responde."""
    global _app
    if _preparado.is_set():
        return True
    try:
        cli.preparar()
    except SystemExit:
        return False
    from asistente.grafo import app
    _app = app
    _preparado.set()
    return True


def ollama_activo() -> bool:
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=6).ok  # con el CPU ocupado tarda más de 2 s
    except requests.RequestException:
        return False


def nueva_sesion() -> str:
    sesion = f"web-{uuid.uuid4().hex[:12]}"
    SESIONES[sesion] = []
    return sesion


class Manejador(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def log_message(self, formato, *args):  # la terminal solo muestra las preguntas, no cada archivo servido
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")  # permite abrir el HTML con doble clic (file://)
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def _json(self, codigo: int, data: dict) -> None:
        cuerpo = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def do_GET(self):
        if self.path.split("?")[0] == "/api/estado":
            return self._json(200, {"ollama": ollama_activo(), "modelo": MODELO, "listo": _preparado.is_set()})
        return super().do_GET()

    def do_POST(self):
        try:
            largo = int(self.headers.get("Content-Length") or 0)
            datos = json.loads(self.rfile.read(largo) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._json(400, {"error": "JSON inválido"})
        ruta = self.path.split("?")[0]
        if ruta == "/api/sesion":
            return self._json(200, {"sesion": nueva_sesion()})
        if ruta == "/api/chat":
            return self._chat(datos)
        return self._json(404, {"error": "ruta no encontrada"})

    def _evento(self, data: dict) -> None:
        self.wfile.write(f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _chat(self, datos: dict) -> None:
        mensaje = str(datos.get("mensaje") or "").strip()[:2000]
        if not mensaje:
            return self._json(400, {"error": "mensaje vacío"})
        sesion = str(datos.get("sesion") or "") or nueva_sesion()
        historial = SESIONES.setdefault(sesion, [])

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            self._evento({"tipo": "inicio", "sesion": sesion})
            if not preparar():
                return self._evento({"tipo": "error", "mensaje": "No hay conexión con Ollama. Ábrelo y vuelve a intentar."})
            with TURNO:
                self._responder(mensaje, historial, sesion)
        except ConnectionError:  # BrokenPipe, Reset o Aborted: la pestaña se cerró o recargó
            print(f"  (la página se cerró antes de terminar la respuesta · {sesion})")

    def _responder(self, mensaje: str, historial: list, sesion: str) -> None:
        from asistente.agentes import NOMBRE_AGENTE

        t0 = time.perf_counter()
        estado = {}
        # Tokens que reporta Ollama (usage_metadata llega en el último fragmento de cada llamada al LLM)
        tokens = {"respuesta": 0, "entrada": 0, "salida": 0}
        try:
            for modo, dato in _app.stream({"mensaje": mensaje, "historial": historial, "sesion": sesion},
                                          stream_mode=["custom", "messages", "values"]):
                if modo == "custom":
                    evento = {"tipo": "fijo", "clase": dato.get("clase", "aviso"), "texto": dato["texto"].strip()}
                    if "datos" in dato:
                        evento["datos"] = dato["datos"]
                    self._evento(evento)
                elif modo == "messages":
                    chunk, meta = dato
                    uso = getattr(chunk, "usage_metadata", None)
                    if uso:  # cuenta todas las llamadas: clasificación + respuesta
                        tokens["entrada"] += uso.get("input_tokens", 0)
                        tokens["salida"] += uso.get("output_tokens", 0)
                        if meta.get("langgraph_node") in cli.NODOS_QUE_RESPONDEN:
                            tokens["respuesta"] += uso.get("output_tokens", 0)
                    if meta.get("langgraph_node") in cli.NODOS_QUE_RESPONDEN and chunk.content:
                        texto = EMOJI.sub("", chunk.content).replace("**", "").replace("#", "")
                        self._evento({"tipo": "token", "texto": texto})
                else:
                    estado = dato
        except requests.RequestException:
            return self._evento({"tipo": "error", "mensaje": "Se perdió la conexión con Ollama. Revisa que siga abierto."})
        except ConnectionError:
            raise  # la página se cerró: lo atiende _chat
        except Exception as e:  # noqa: BLE001  (la página debe enterarse de cualquier fallo)
            print(f"  ERROR: {e!r}")
            return self._evento({"tipo": "error", "mensaje": f"Ocurrió un error al responder: {e}"})
        segundos = round(time.perf_counter() - t0, 1)
        agente = next((n for n in reversed(estado.get("ruta", [])) if n in NOMBRE_AGENTE), "")
        nombre = NOMBRE_AGENTE.get(agente, "-")
        respuesta = estado.get("respuesta", "")
        historial.append((mensaje, respuesta))
        print(f"[{time.strftime('%H:%M:%S')}] {sesion} · {nombre} · {segundos} s · "
              f"{tokens['respuesta']} tokens · {mensaje[:70]}")
        from asistente import herramientas as H  # registro en tiempo real de este paciente (solo lo que dijo)
        glucosas = [{"mg_dl": g["mg_dl"], "fecha": g["fecha"]} for g in H.historial_glucosa(sesion, n=12)]
        self._evento({"tipo": "fin", "agente": nombre, "segundos": segundos, "respuesta": respuesta, "tokens": tokens,
                      "perfil": H.obtener_perfil(sesion), "glucosas": glucosas})


class Servidor(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], ConnectionError):
            return  # el navegador cerró la conexión (recarga, pestaña cerrada): no es un error del servidor
        super().handle_error(request, client_address)


def ip_local() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.168.0.1", 80))  # no envía nada: solo elige la interfaz de red
            return s.getsockname()[0]
    except OSError:
        return "tu-ip-local"


def main() -> None:
    p = argparse.ArgumentParser(description="Interfaz web del asistente LLM Diabetes")
    p.add_argument("--puerto", type=int, default=5500)  # debe coincidir con PUERTO_SERVIDOR de interfaz/asistente.js
    p.add_argument("--red", action="store_true", help="permitir conexiones de otros equipos en la misma red WiFi")
    args = p.parse_args()

    sys.stdout.reconfigure(line_buffering=True)  # el log de cada pregunta aparece al momento
    host = "0.0.0.0" if args.red else "127.0.0.1"
    servidor = Servidor((host, args.puerto), Manejador)
    print("LLM Diabetes · cargando modelo (la primera vez tarda ~20 s)...", flush=True)
    if not preparar():
        print("AVISO: Ollama no responde. Ábrelo; el servidor lo reintentará en la primera pregunta.")
    print(f"Listo ({MODELO}). Abre en el navegador:  http://localhost:{args.puerto}")
    if args.red:
        print(f"En la misma WiFi:                       http://{ip_local()}:{args.puerto}")
        print("  (cualquier equipo de esa red podrá usar el asistente mientras el servidor esté abierto)")
    print("Ctrl+C para detener.\n")
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
    finally:
        servidor.server_close()


if __name__ == "__main__":
    main()
