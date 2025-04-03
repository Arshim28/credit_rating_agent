from typing import Optional, Type, Any, Literal, get_type_hints
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.managed import RemainingSteps
from langchain_core.messages import HumanMessage


class MessagesWithSteps(MessagesState):
    remaining_steps: RemainingSteps


def end_or_reflect(state: MessagesWithSteps) -> Literal[END, "graph"]:
    """Determine whether to continue or end the reflection loop."""
    # Always check remaining steps first
    if state["remaining_steps"] < 2:
        return END
        
    # If no messages, end
    if len(state["messages"]) == 0:
        return END
        
    # Check if we have any reports left to process
    if "current_report_index" in state and "reports_data" in state:
        if state["current_report_index"] >= len(state["reports_data"]):
            # Processed all reports
            if "trend_analysis" in state and state["trend_analysis"]:
                # Finalization happened, we're done
                return END
                
    # If last message is from human, continue with main graph
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        return "graph"
    else:
        return END


def create_reflection_graph(
    graph: CompiledStateGraph,
    reflection: CompiledStateGraph,
    state_schema: Optional[Type[Any]] = None,
    config_schema: Optional[Type[Any]] = None,
) -> StateGraph:
    _state_schema = state_schema or graph.builder.schema

    annotations = getattr(_state_schema, "__annotations__", {})
    
    if "remaining_steps" in annotations:
        raise ValueError(
            "Has key 'remaining_steps' in state_schema, this shadows a built in key"
        )

    if "messages" not in annotations:
        raise ValueError("Missing required key 'messages' in state_schema")

    # Dynamically create a new class by inheriting from the state schema
    class StateSchema(_state_schema):
        remaining_steps: RemainingSteps

    rgraph = StateGraph(StateSchema, config_schema=config_schema)
    rgraph.add_node("graph", graph)
    rgraph.add_node("reflection", reflection)
    rgraph.add_edge(START, "graph")
    rgraph.add_edge("graph", "reflection")
    rgraph.add_conditional_edges("reflection", end_or_reflect)
    return rgraph


__all__ = ["MessagesWithSteps", "end_or_reflect", "create_reflection_graph"]