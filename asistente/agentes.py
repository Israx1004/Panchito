"""Agentes (Tarea 5): clasificación, Supervisor y agentes especializados.

Flujo por pregunta (una sola respuesta generada por el LLM):
  clasificar   Zero-shot + Tagging (1 llamada corta al LLM), alarma por reglas,
               Clustering (grupo de preguntas frecuentes más cercano, por embeddings)
               Calculadora nutrimental: alimentos y cantidades del mensaje (tabla local, sin LLM)
  supervisor   elige el agente sin llamar al LLM
  agente_*     usa sus herramientas (RAG, BD, búsqueda, calculadoras) y genera la respuesta en streaming

El texto fijo (aviso de emergencia, fuera de tema) se emite sin el modelo: no depende de que el LLM acierte.
"""
from __future__ import annotations

import re
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer

from asistente import herramientas as H
from asistente import indice
from asistente import nutricion as N
from asistente.tecnicas import Clasificacion, clasificar, hay_senal_alarma
from config import AVISO_EMERGENCIA, MENSAJE_FUERA_DE_TEMA, PROMPT_BASE, limpiar, llm, norm

MAX_EVIDENCIA = 450   # caracteres por fragmento de evidencia
_llm = None


def _modelo():
    global _llm
    if _llm is None:
        _llm = llm(temperature=0.3, num_predict=512)
    return _llm


class EstadoConversacion(TypedDict, total=False):
    mensaje: str
    sesion: str                        # id de la conversación: perfil y mediciones son de esta sesión
    perfil: dict                       # tipo, peso_kg, estatura_m: solo lo que el usuario dijo
    historial: list[tuple[str, str]]   # (paciente, asistente) de turnos anteriores
    vector: Any                        # embedding de la pregunta: se calcula una vez (clustering + RAG)
    clasificacion: Clasificacion
    grupo: dict
    alarma: bool
    alimentos: list[dict]              # alimentos y gramos detectados por la calculadora nutrimental
    pide_calculo: bool                 # el usuario pide calorías, carbohidratos, proteína, fibra...
    siguiente: str
    ruta: list[str]
    contexto: dict[str, Any]
    respuesta: str


def _emitir(texto: str, clase: str = "aviso", **extra) -> None:
    """Texto fijo que se muestra de inmediato (no pasa por el LLM). `clase` le dice a la interfaz web cómo mostrarlo:
    tabla, emergencia, estimado, fuera; la terminal solo imprime `texto`."""
    get_stream_writer()({"texto": texto, "clase": clase, **extra})


def _habla_de_diabetes(estado: EstadoConversacion) -> bool:
    texto = norm(" ".join([estado["mensaje"], *(p for p, _ in estado.get("historial", []))]))
    return bool(re.search(r"diabet|glucosa|insulin|glucemi|metformina", texto))


def _sesion(estado: EstadoConversacion) -> str:
    return estado.get("sesion") or "local"


def _datos_perfil(perfil: dict, diabetes: bool) -> list[str]:
    """Solo datos que el usuario dio; si el tema es diabetes y no dijo el tipo, se avisa al LLM que no lo asuma."""
    datos = []
    if tipo := perfil.get("tipo"):
        datos.append(f"tiene diabetes {'gestacional' if tipo == 'gestacional' else 'tipo ' + tipo}")
    elif diabetes:
        datos.append("tipo de diabetes: no lo ha dicho (no asumas tipo 1 ni tipo 2)")
    if peso := perfil.get("peso_kg"):
        datos.append(f"peso {peso:g} kg")
    if est := perfil.get("estatura_m"):
        datos.append(f"estatura {est:g} m")
    if peso and est:
        datos.append(f"IMC {H.calcular_imc(peso, est)}")
    return datos


def _responder(rol: str, estado: EstadoConversacion, evidencia: list[str], diabetes: bool = False) -> str:
    c = estado["clasificacion"]
    datos = [f"glucosa {c.glucosa} mg/dl" if c.glucosa else "",
             *_datos_perfil(estado.get("perfil", {}), diabetes or _habla_de_diabetes(estado)),
             f"síntomas: {', '.join(c.sintomas)}" if c.sintomas else ""]
    previo = estado.get("historial", [])[-1:]
    partes = [f"Antes el usuario dijo: {p[0]} | Respondiste: {r[:300]}" for p, r in previo]
    partes.append(f"Usuario: {estado['mensaje']}")
    if any(datos):
        partes.append("Datos: " + "; ".join(d for d in datos if d))
    partes.append("Evidencia:\n" + "\n".join(f"- {e[:MAX_EVIDENCIA]}" for e in evidencia))
    salida = _modelo().invoke([SystemMessage(f"{PROMPT_BASE}\nRol: {rol}."), HumanMessage("\n".join(partes))])
    return limpiar(salida.content)


def _textos(docs: list[dict]) -> list[str]:
    # De las preguntas curadas solo va la respuesta: la pregunta es de otra persona (p. ej. "¿dejo la metformina?")
    # y el LLM la tomaba como dato del usuario.
    return [d["texto"].split(" R: ", 1)[-1] if d.get("fuente") == "qa_curado" else d["texto"] for d in docs]


# ---------- Nodos ----------
def nodo_clasificar(estado: EstadoConversacion) -> EstadoConversacion:
    msg = estado["mensaje"]
    # El mensaje anterior solo se usa en seguimientos cortos ("¿y si como pan?"); si no, contamina la ruta.
    corto = len(msg.split()) <= 5 and msg.lower().lstrip("¿ ").startswith(("y ", "e ", "o ", "entonces", "pero"))
    previo = estado["historial"][-1][0] if estado.get("historial") and corto else ""
    vector = indice.vectorizar(msg)
    c = clasificar(msg, previo)
    # Perfil del paciente real: se actualiza con lo que dice y se recuerda en la sesión.
    nuevo = H.extraer_perfil(msg)
    if c.tipo and "tipo" not in nuevo:
        nuevo["tipo"] = str(c.tipo)
    H.actualizar_perfil(_sesion(estado), **nuevo)
    alimentos = N.extraer_alimentos(msg)
    if c.categoria == "fuera" and alimentos:  # menciona un alimento de la tabla: sí es del dominio
        c.categoria = "dieta"
    # Seguimiento de una medición ("ahora salió en 320") sin repetir la palabra glucosa: cuenta como glucosa
    # si el mensaje anterior del usuario hablaba de glucosa y este no habla de comida.
    if not c.glucosa and not alimentos and estado.get("historial"):
        anterior = norm(estado["historial"][-1][0])
        m = re.search(r"\b(\d{2,3})\b", norm(msg))
        if m and 20 <= int(m.group(1)) <= 700 and re.search(r"glucosa|azucar|glucometro|mg|dl", anterior):
            c.glucosa = int(m.group(1))
            if c.categoria not in ("emergencia", "medicacion"):
                c.categoria = "seguimiento"
    return {"vector": vector, "clasificacion": c, "grupo": indice.grupo(vector), "alarma": hay_senal_alarma(msg),
            "perfil": H.obtener_perfil(_sesion(estado)),
            "alimentos": alimentos, "pide_calculo": bool(N.PIDE_CALCULO.search(N.texto_normal(msg))),
            "ruta": ["clasificar"], "contexto": {}}


MEDICACION = re.compile(r"insulin|dosis|unidades de|pastilla|medicament|metformina|glibenclamida")


def nodo_supervisor(estado: EstadoConversacion) -> EstadoConversacion:
    c = estado["clasificacion"]
    if estado.get("alarma") or c.categoria == "emergencia":
        siguiente = "agente_emergencia"
    elif c.categoria == "medicacion" or MEDICACION.search(norm(estado["mensaje"])):  # dosis antes que calorías
        siguiente = "agente_medicacion"
    elif c.glucosa:  # glucosa reportada: se registra y puede escalar a Emergencias
        siguiente = "agente_monitoreo"
    elif estado.get("alimentos") and (estado.get("pide_calculo")
                                     or any(not a["estimado"] for a in estado["alimentos"])):
        siguiente = "agente_calculo"  # pide nutrientes o da cantidades: la calculadora responde con cifras
    elif c.categoria == "fuera":
        siguiente = "fuera_de_tema"
    elif c.categoria == "dieta" or (estado.get("pide_calculo") and c.categoria != "seguimiento"):
        siguiente = "agente_nutricion"
    elif c.categoria in ("sintomas", "seguimiento"):  # datos del propio paciente
        siguiente = "agente_monitoreo"
    else:  # preguntas generales sobre diabetes o salud
        siguiente = "agente_educacion"
    return {"siguiente": siguiente, "ruta": estado["ruta"] + ["supervisor"]}


def fuera_de_tema(estado: EstadoConversacion) -> EstadoConversacion:
    _emitir(MENSAJE_FUERA_DE_TEMA, "fuera")
    return {"respuesta": MENSAJE_FUERA_DE_TEMA, "ruta": estado["ruta"] + ["fuera_de_tema"]}


def agente_educacion(estado: EstadoConversacion) -> EstadoConversacion:
    # Clustering: el tema del grupo de preguntas frecuentes más cercano guía la búsqueda en el RAG.
    guia = indice.TEMA_POR_CATEGORIA.get(estado["grupo"]["tema"])
    evidencia = _textos(H.rag(estado["vector"], guia=guia))
    return {"respuesta": _responder("Agente de Educación: explica conceptos de salud y diabetes", estado, evidencia),
            "ruta": estado["ruta"] + ["agente_educacion"]}


GUIAS_NUTRICION = ("nutricion", "macronutrientes", "dieta")


def agente_nutricion(estado: EstadoConversacion) -> EstadoConversacion:
    evidencia = _textos(H.rag(estado["vector"], guia=GUIAS_NUTRICION))
    evidencia += [N.ficha(a["alimento"]) for a in estado.get("alimentos", [])[:3]]
    if estado.get("pide_calculo") and not estado.get("alimentos"):
        evidencia.append("[regla] Ese alimento no está en la tabla nutrimental: no inventes cifras; "
                         "sugiere revisar la etiqueta o pesar la porción.")
    rol = "Agente de Nutrición"  # peso/estatura/IMC solo entran (vía Datos) si el usuario los dio
    if estado.get("alimentos"):
        rol += "; si das calorías o nutrientes, di claramente que son solo un estimado"
    respuesta = _responder(rol, estado, evidencia)
    if estado.get("alimentos") and re.search(r"\d+\s*(kcal|calorias|g|gramos)\b", norm(respuesta)):
        respuesta = _asegurar_estimado(respuesta)  # solo si la respuesta dio cifras
    return {"respuesta": respuesta, "ruta": estado["ruta"] + ["agente_nutricion"]}


def _asegurar_estimado(respuesta: str) -> str:
    """Red de seguridad: si el LLM no aclaró que las cifras son estimadas, se agrega un aviso fijo."""
    if "estimad" in norm(respuesta) or "aproximad" in norm(respuesta):
        return respuesta
    _emitir("\n" + N.AVISO_ESTIMADO, "estimado")
    return f"{respuesta}\n{N.AVISO_ESTIMADO}"


def agente_calculo(estado: EstadoConversacion) -> EstadoConversacion:
    """Calculadora nutrimental: la tabla con cifras es fija (sin LLM); el LLM solo la interpreta."""
    calc = N.calcular(estado["alimentos"])
    tabla = N.formatear(calc)
    _emitir(tabla, "tabla", datos=calc)
    evidencia = [N.resumen(calc)] + _textos(H.rag(estado["vector"], guia=("macronutrientes", "nutricion"), k=2))
    rol = ("Agente de Cálculo nutrimental: el usuario ya ve la tabla con las cifras; no la repitas completa. "
           "Interpreta los totales en 2 o 3 oraciones, recalca que las cifras son solo un estimado "
           "y da recomendaciones breves. Si el usuario mencionó algo que no aparece en la tabla, di que no tienes "
           "su dato; no calcules tú sus calorías ni nutrientes")
    respuesta = _asegurar_estimado(_responder(rol, estado, evidencia))
    return {"respuesta": f"{tabla}\n{respuesta}", "ruta": estado["ruta"] + ["agente_calculo"],
            "contexto": {"calculo": calc}}


def agente_medicacion(estado: EstadoConversacion) -> EstadoConversacion:
    evidencia = _textos(H.rag(estado["vector"], guia="insulina"))
    if ficha := H.buscar_info_medicamento(estado["mensaje"]):
        evidencia.insert(0, f"[búsqueda] {ficha}")
    evidencia.append("[regla] No indicar ni calcular dosis; remitir al médico tratante.")
    return {"respuesta": _responder("Agente de Medicación", estado, evidencia, diabetes=True),
            "ruta": estado["ruta"] + ["agente_medicacion"]}


def agente_monitoreo(estado: EstadoConversacion) -> EstadoConversacion:
    """Monitoreo en tiempo real: solo las mediciones que este paciente reportó en esta sesión."""
    actual = estado["clasificacion"].glucosa
    if actual:
        H.registrar_glucosa(float(actual), _sesion(estado))
    valores = [int(h["mg_dl"]) for h in H.historial_glucosa(_sesion(estado), n=5)]
    # Colaboración: un valor actual fuera de rango escala a Emergencias (que genera la única respuesta).
    if actual and (actual < 70 or actual > 300):
        return {"siguiente": "agente_emergencia", "ruta": estado["ruta"] + ["agente_monitoreo"],
                "contexto": {"historial": valores}}
    evidencia = _textos(H.rag(estado["vector"], guia="seguimiento"))
    if len(valores) >= 3:
        tendencia = ("ascendente" if valores[-1] > valores[0] + 30 else
                     "descendente" if valores[-1] < valores[0] - 30 else "estable")
        evidencia.append(f"[BD] Mediciones que el paciente reportó en esta conversación (mg/dl, de la más antigua "
                         f"a la más reciente): {valores}, tendencia {tendencia}")
    elif valores:
        evidencia.append(f"[BD] Mediciones que el paciente reportó en esta conversación (mg/dl): {valores}. "
                         "No hay suficientes para una tendencia: no hables de mediciones anteriores; "
                         "sugiere registrar varias en distintos momentos.")
    else:
        evidencia.append("[BD] El paciente no ha reportado mediciones de glucosa en esta conversación: "
                         "no inventes valores; pídele su medición si la necesitas.")
    return {"respuesta": _responder("Agente de Monitoreo", estado, evidencia, diabetes=True),
            "ruta": estado["ruta"] + ["agente_monitoreo"], "siguiente": "fin"}


def agente_emergencia(estado: EstadoConversacion) -> EstadoConversacion:
    _emitir(AVISO_EMERGENCIA, "emergencia")  # fijo e inmediato, antes de generar
    evidencia = _textos(H.rag(estado["vector"], guia="emergencias", k=2))
    if valores := estado.get("contexto", {}).get("historial"):
        evidencia.append(f"[BD] Mediciones que el paciente reportó en esta conversación (mg/dl): {valores}")
    glucosa = estado["clasificacion"].glucosa
    if glucosa and glucosa > 300:
        evidencia.insert(0, f"[regla] {glucosa} mg/dl es HIPERglucemia (glucosa muy alta): no hables de hipoglucemia "
                            "ni recomiendes dar azúcar.")
    elif glucosa and glucosa < 70:
        evidencia.insert(0, f"[regla] {glucosa} mg/dl es HIPOglucemia (glucosa baja): si la persona está consciente y "
                            "puede tragar, sigue su plan para glucosa baja; si no puede tragar, nada por la boca.")
    s = H.buscar_servicios("urgencias")[0]
    evidencia.append(f"[servicios] {s['nombre']}: {s['contacto']}")
    respuesta = _responder("Agente de Emergencias: prioriza la seguridad", estado, evidencia)
    return {"respuesta": f"{AVISO_EMERGENCIA}\n{respuesta}", "ruta": estado["ruta"] + ["agente_emergencia"]}


AGENTES = {"agente_educacion": agente_educacion, "agente_nutricion": agente_nutricion, "agente_calculo": agente_calculo,
           "agente_medicacion": agente_medicacion,
           "agente_monitoreo": agente_monitoreo, "agente_emergencia": agente_emergencia,
           "fuera_de_tema": fuera_de_tema}
NOMBRE_AGENTE = {"agente_educacion": "Educación", "agente_nutricion": "Nutrición", "agente_calculo": "Cálculo nutrimental",
                 "agente_medicacion": "Medicación", "agente_monitoreo": "Monitoreo",
                 "agente_emergencia": "Emergencias", "fuera_de_tema": "Filtro de tema"}
