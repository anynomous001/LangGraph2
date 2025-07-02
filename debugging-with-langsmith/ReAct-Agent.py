from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import tools_condition,ToolNode
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage



from typing import Annotated
from typing_extensions import TypedDict
from IPython.display import display,Image
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()


os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"


tavily = TavilySearchResults()

tavily.invoke({"query": "recent developments in AI field"})


#custom tools
def addition(State: dict) -> dict:
    """Function to add two numbers."""
    a = State['a']
    b = State['b']
    result = a + b
    return {'result': result}


def subtraction(State: dict):
    """Function to subtract two numbers."""
    a = State['a']
    b = State['b']
    result = a - b
    return {'result': result}


def multiplication(State: dict) -> dict:
    """Function to multiply two numbers."""
    a = State['a']
    b = State['b']
    result = a * b
    return {'result': result}

def division(State: dict) -> dict:
    """Function to divide two numbers."""
    a = State['a']
    b = State['b']
    if b == 0:
        return {'error': 'Division by zero is not allowed.'}
    result = a / b
    return {'result': result}


def current_weather(State: dict) -> dict:
    """Function to get the current weather of a city."""
    import requests

    city = State['city']
    try:
        url = f"https://wttr.in/{city}?format=3"
        response = requests.get(url)
        if response.status_code == 200:
            weather = response.text.strip()
            return {'weather': weather}
        else:
            return {'weather': f"Could not fetch weather for {city} (status code: {response.status_code})"}
    except Exception as e:
        return {'weather': f"Error fetching weather: {e}"}      

from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun

arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=3,doc_content_chars_max=600)
wikipedia_api_wrapper = WikipediaAPIWrapper(top_k_results=3,doc_content_chars_max=600)

arxiv = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)
wikipedia = WikipediaQueryRun(api_wrapper=wikipedia_api_wrapper)

#tools

tools = [arxiv, wikipedia, tavily,multiplication, division, addition, subtraction, current_weather]

llm = ChatGroq(model="Gemma2-9b-It")

# Bind the tools to the LLM

llm_with_tools = llm.bind_tools(tools)





class State(TypedDict):
    messages :Annotated[list[AnyMessage],add_messages]




def tool_function(State:State) -> dict:
    """    Function to invoke the LLM with tools and return the messages.
    Args:
        State (State): The state containing the messages.
    Returns:
        dict: A dictionary containing the messages after invoking the LLM with tools.

    """   
     # Invoke the LLM with tools and return the messages
    return {'messages':llm_with_tools.invoke(State['messages'])}


def make_graph():
    """Function to create the state graph."""
    graph = StateGraph(State)

    graph.add_node("tool_function",tool_function)
    graph.add_node("tools",ToolNode(tools))


    graph.add_edge(START,"tool_function")
    graph.add_conditional_edges("tool_function",tools_condition)
    graph.add_edge("tools","tool_function")


    # memory = MemorySaver()
    # Compile the graph with memory checkpointer
    graph_builder = graph.compile()


    # # config = {"configurable":{"thread_id":"3"}}
    # messages=graph_builder.invoke({"messages": [HumanMessage(" tell me current temapertaure of kolkata")]},config=config)

    # for message in messages['messages']:
    #     message.pretty_print()  # Print the message content

    return graph_builder


agent = make_graph()