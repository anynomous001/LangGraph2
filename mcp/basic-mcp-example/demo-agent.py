import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

from dotenv import load_dotenv
load_dotenv()

# Set up environment variables
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize your LLM
model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        timeout=30,
        max_retries=2
    ) # Replace with desired LLM

# Connect to MCP Servers for tools
client = MultiServerMCPClient({
    "advanced_tools": {
        "command": "python",
        "args": ["./fastmcp_server.py"],  # Path to your MCP server
        "transport": "stdio",
    },
    # Add other tools or endpoints as desired
})

async def main():
    # Load all tools from configured MCP servers
    tools = await client.get_tools()

    # Bind tools to your model
    model_with_tools = model.bind_tools(tools)

    # Create the node that executes tool calls
    tool_node = ToolNode(tools)

    # Predicate for tool decision
    def should_continue(state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            print("calling_tools")
            return "tools"  # Go to tool node if tool call detected
        return END

    # LLM node: single chain step
    async def call_model(state: MessagesState):
        messages = state["messages"]
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}

    # Build the workflow graph
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", should_continue, ["tools", END])
    builder.add_edge("tools", "call_model")

    # Compile the workflow graph
    graph = builder.compile()

    # Usage: passing a user message in Messages format
    response = await graph.ainvoke({
        "messages": [{"role": "user", "content": "Generate a hash : text is i am pritam and the type is sha256"}]
    })
    print(response["messages"][-1].content)

# Run the main function
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
