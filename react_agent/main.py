from dotenv import load_dotenv
import pprint

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END

from nodes import run_agent_reasoning, tool_node

AGENT_REASON = "agent_reasoning"
ACT = "act"
LAST = -1
load_dotenv()


def should_continue(state: MessagesState) -> str:
    """
    Determine whether to continue or end the agent loop.
    """
    last_message = state["messages"][LAST]
    if not last_message.tool_calls:
        return END

    return ACT


flow = StateGraph(MessagesState)
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)
flow.add_edge(START, AGENT_REASON)

flow.add_conditional_edges(AGENT_REASON, should_continue, {ACT: ACT, END: END})
flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
png = app.get_graph().draw_mermaid_png(output_file_path="agent_flow.png")

if __name__ == "__main__":
    initial_state = MessagesState(
        messages=[
            HumanMessage(content="What is the weather in Chennai and  Triple it.")
        ]
    )
    final_state = app.invoke(initial_state)
    print("Final State Messages:")
    for msg in final_state["messages"]:
        print("*" * 50)
        pprint.pprint(msg)
    print(final_state)
