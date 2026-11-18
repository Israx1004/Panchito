const Asistente = (() => {
  // Puerto
  const PUERTO_SERVIDOR = "5500";
  // Si la página la sirve otro programa (Live Server en 5500) o se abre con doble clic, las llamadas van a servidor.py.
  const BASE = location.port === PUERTO_SERVIDOR ? "" : `http://${location.hostname || "localhost"}:${PUERTO_SERVIDOR}`;
  const CLAVE = "llm-diabetes-sesion";

  const leer = () => { try { return sessionStorage.getItem(CLAVE); } catch { return null; } };
  const guardar = (s) => { try { sessionStorage.setItem(CLAVE, s); } catch { /* sin almacenamiento: dura hasta recargar */ } };
  let sesion = leer();


  async function estado() {
    try {
      const r = await fetch(`${BASE}/api/estado`);
      return await r.json();
    } catch {
      return { ollama: false, modelo: null, listo: false, servidor: false };
    }
  }

  async function nuevaConversacion() {
    const r = await fetch(`${BASE}/api/sesion`, { method: "POST" });
    sesion = (await r.json()).sesion;
    guardar(sesion);
    return sesion;
  }

  async function enviar(mensaje, { alFijo = () => {}, alTexto = () => {}, alFin = () => {}, alError = () => {} } = {}) {
    let respuesta;
    try {
      respuesta = await fetch(`${BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mensaje, sesion }),
      });
    } catch {
      alError("No se pudo conectar con el servidor");
      return;
    }
    if (!respuesta.ok || !respuesta.body) {
      alError(`El servidor respondió con un error (${respuesta.status}).`);
      return;
    }

    const lector = respuesta.body.getReader();
    const decodificador = new TextDecoder();
    let pendiente = "";
    let terminado = false;
    for (;;) {
      const { value, done } = await lector.read();
      if (done) break;
      pendiente += decodificador.decode(value, { stream: true });
      const bloques = pendiente.split("\n\n");
      pendiente = bloques.pop();
      for (const bloque of bloques) {
        const linea = bloque.split("\n").find((l) => l.startsWith("data: "));
        if (!linea) continue;
        const ev = JSON.parse(linea.slice(6));
        if (ev.tipo === "inicio") { sesion = ev.sesion; guardar(sesion); }
        else if (ev.tipo === "fijo") alFijo(ev);
        else if (ev.tipo === "token") alTexto(ev.texto);
        else if (ev.tipo === "fin") { terminado = true; alFin(ev); }
        else if (ev.tipo === "error") { terminado = true; alError(ev.mensaje); }
      }
    }
    if (!terminado) alError("La respuesta se interrumpió. Intenta de nuevo.");
  }

  return { estado, enviar, nuevaConversacion, get sesion() { return sesion; } };
})();
