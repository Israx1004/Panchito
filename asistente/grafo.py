"""Orquestación con LangGraph (Tarea 5): grafo de estados explícito.

START -> clasificar -> supervisor -> {cálculo nutrimental | nutrición | educación | medicación | monitoreo |
                                      emergencias | fuera de tema} -> END
Monitoreo escala a Emergencias si la glucosa reportada está fuera de rango.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from asistente.agentes import AGENTES, EstadoConversacion, nodo_clasificar, nodo_supervisor


def construir_grafo():
    g = StateGraph(EstadoConversacion)
    g.add_node("clasificar", nodo_clasificar)
    g.add_node("supervisor", nodo_supervisor)
    for nombre, fn in AGENTES.items():
        g.add_node(nombre, fn)
    g.add_edge(START, "clasificar")
    g.add_edge("clasificar", "supervisor")
    g.add_conditional_edges("supervisor", lambda e: e["siguiente"], list(AGENTES))
    g.add_conditional_edges("agente_monitoreo", lambda e: e["siguiente"],
                            {"agente_emergencia": "agente_emergencia", "fin": END})
    for nombre in AGENTES:
        if nombre != "agente_monitoreo":
            g.add_edge(nombre, END)
    return g.compile()


app = construir_grafo()
