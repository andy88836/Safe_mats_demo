"""LangGraph state machine assembly for MOF screening pipeline."""
from langgraph.graph import StateGraph, END

from agent.state import ScreeningState
from agent import nodes


def build_graph():
    g = StateGraph(ScreeningState)

    g.add_node("ingest_cif", nodes.ingest_cif)
    g.add_node("compute_descriptor", nodes.compute_descriptor)
    g.add_node("predict_adsorption", nodes.predict_adsorption)
    g.add_node("extract_linker", nodes.extract_linker)
    g.add_node("predict_toxicity", nodes.predict_toxicity)
    g.add_node("apply_safety_rules", nodes.apply_safety_rules)
    g.add_node("score_and_explain", nodes.score_and_explain)

    g.set_entry_point("ingest_cif")
    g.add_edge("ingest_cif", "compute_descriptor")
    g.add_edge("compute_descriptor", "predict_adsorption")
    g.add_edge("predict_adsorption", "extract_linker")
    g.add_edge("extract_linker", "predict_toxicity")
    g.add_edge("predict_toxicity", "apply_safety_rules")
    g.add_edge("apply_safety_rules", "score_and_explain")
    g.add_edge("score_and_explain", END)

    return g.compile()


screening_app = build_graph()
