from dotenv import load_dotenv


from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pprint import pprint
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

load_dotenv()


def fancy_box(text):
    length = len(text)
    print("╔" + "═" * (length + 2) + "╗")
    print("║ " + text + " ║")
    print("╚" + "═" * (length + 2) + "╝")


@tool
def modernise(num: float) -> float:
    """
    param num: a number to modernise
    returns: the modernise of the input number
    """
    return float(num) * 3


toolt = [TavilySearch(max_results=1), modernise]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(toolt)


LAST = -1
load_dotenv()
SYSYEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions.
"""


def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """
    response = llm.invoke(
        [{"role": "system", "content": SYSYEM_MESSAGE}, *state["messages"]]
    )
    return {"messages": [response]}


def should_continue(state: MessagesState) -> str:
    """
    Determine whether to continue or end the agent loop.
    """
    last_message = state["messages"][LAST]
    # pprint(last_message.dict())
    if not last_message.tool_calls:
        fancy_box("END")
        return END
    elif last_message.tool_calls[0]["name"] == "tavily_search":
        fancy_box("TAVILY SEARCH TOOL CALLED")
        return TOOL1
    fancy_box("END TRIPLE TOOL CALLED")
    return TOOL2


AGENT_REASON = "agent_reasoning"
TOOL1 = "act_tavily"
TOOL2 = "act_calculator"

flow = StateGraph(MessagesState)

tool_node_search = ToolNode([TavilySearch(max_results=1)])
tool_node_triple = ToolNode([modernise])

flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.add_node(TOOL1, tool_node_search)
flow.add_node(TOOL2, tool_node_triple)

flow.set_entry_point(AGENT_REASON)

flow.add_conditional_edges(
    AGENT_REASON, should_continue, {TOOL1: TOOL1, TOOL2: TOOL2, END: END}
)
flow.add_edge(TOOL1, AGENT_REASON)
flow.add_edge(TOOL2, AGENT_REASON)

app = flow.compile()
png = app.get_graph().draw_mermaid_png(output_file_path="agent_flow2.png")

if __name__ == "__main__":
    initial_state = MessagesState(
        messages=[
            HumanMessage(
                content="What is the weather in Chennai and  modernise the temparature."
            )
        ]
    )
    final_state = app.invoke(initial_state)
    print("Final State Messages:")
    for msg in final_state["messages"]:
        print("*" * 50)
        pprint(msg.dict())
