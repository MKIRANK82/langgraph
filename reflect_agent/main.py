from typing import TypedDict, Annotated

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chains import generate_chain, reflect_chain


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"

def fancy_box(text):
    length = len(text)
    print("╔" + "═"*(length + 2) + "╗")
    print("║ " + text + " ║")
    print("╚" + "═"*(length + 2) + "╝")

    with open("reflect_agent/logs.txt", "a") as f:
        f.write(text + "\n"*5)



def generation_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessageGraph):
    last_message = state["messages"][-1]
    fancy_box(last_message.content)
    res = reflect_chain.invoke({"messages": state["messages"]})
    fancy_box(res.content)
    return {"messages": [HumanMessage(content=res.content)]}


builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue,{END: END, REFLECT: REFLECT})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())

graph.get_graph().draw_mermaid_png(output_file_path="reflect_agent/agent_flow.png")



if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = {
        "messages": [
            HumanMessage(
                content="""
Generate a engaging tweet about the importance of AI, but the wave if AI is making peole implement non AI changes as AI.
There should be a real usecase for AI and not for fancy AI work.
Most works we see are either acheived via work flow or simple llm and not an agentic approach.
I see many examples of analysis past history for few aggregators using agents which is like killing an ant with a sledge hammer.

                                  """
            )
        ]
    }
    response = graph.invoke(inputs)
    print(response)