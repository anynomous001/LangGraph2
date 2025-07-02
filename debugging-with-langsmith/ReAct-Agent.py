from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from typing import Annotated
from typing_extensions import TypedDict
from IPython.display import display, Image
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

llm = ChatGroq(model="Gemma2-9b-It")



def make_graph():
    @tool
    def addition(a: int, b: int) -> int:
        """Function to add two numbers."""
        return a + b

    @tool
    def subtraction(a: int, b: int) -> int:
        """Function to subtract two numbers."""
        return a - b

    @tool
    def multiplication(a: int, b: int) -> int:
        """Function to multiply two numbers."""
        return a * b
    
    @tool
    def division(a: float, b: float) -> float:
        """Function to divide two numbers."""
        if b == 0:
            raise ValueError('Division by zero is not allowed.')
        return a / b

    @tool
    def current_weather(city: str) -> str:
        """Function to get the current weather of a city."""
        import requests
        try:
            url = f"https://wttr.in/{city}?format=3"
            response = requests.get(url)
            if response.status_code == 200:
                return response.text.strip()
            else:
                return f"Could not fetch weather for {city} (status code: {response.status_code})"
        except Exception as e:
            return f"Error fetching weather: {e}"

    # Initialize tools
    # @tool
    # tavily = TavilySearchResults()
    
 
    arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=600)
    wikipedia_api_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=600)

    arxiv = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)
    wikipedia = WikipediaQueryRun(api_wrapper=wikipedia_api_wrapper)

    # Create tool list
    tools = [arxiv, wikipedia, tavily, multiplication, division, addition, subtraction, current_weather]
    tool_node = ToolNode(tools)

    # Initialize LLM
    llm_with_tools = llm.bind_tools(tools)

    

    def agent_node(state: State) -> dict:
        """Main agent function that calls the LLM with tools."""
        return {'messages': [llm_with_tools.invoke(state['messages'])]}

    def should_continue(state: State):
        """Function to check if the conversation should continue."""
        last_message = state['messages'][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return 'tools'
        else:
            return END

    # Create the graph
    graph = StateGraph(State)
    
    # Add nodes
    graph.add_node("agent", agent_node)  # Renamed for clarity
    graph.add_node("tools", tool_node)
    
    # Add edges
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")  # This creates the loop back to agent
    
    # Compile the graph
    return graph.compile()

# Create the agent
agent = make_graph()

# To visualize the graph structure, you can use:
# display(Image(agent.get_graph().draw_mermaid_png()))