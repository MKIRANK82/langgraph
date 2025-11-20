from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.tools import StructuredTool


tavily_tool = TavilySearch(max_results=5)


@tool
def run_queries(search_queries: list[str], **kwargs) -> list[str]:
    """Run the generated queries."""
    return tavily_tool.batch([{"query": query} for query in search_queries])


print(run_queries)

execute_tools = run_queries

response = execute_tools.invoke(
    {
        "search_queries": [
            "What is the capital of France?",
            "Who is the president of the United States?",
        ]
    }
)
print(response)
