#!/usr/bin/env python3
"""
Simple MCP Client Example
Connects to MCP server and calls tools
"""

import asyncio
import subprocess
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    """Main client function"""
    
    # Server parameters - adjust path to your server script
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"]  # Path to your server script
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            
            # Initialize the session
            await session.initialize()
            
            print("Connected to MCP server!")
            print("-" * 50)
            
            # List available tools
            tools_result = await session.list_tools()
            print(f"Available tools: {len(tools_result.tools)}")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")
            print()
            
            # Test echo tool
            print("Testing echo tool:")
            echo_result = await session.call_tool(
                "echo", 
                {"text": "Hello from MCP client!"}
            )
            for content in echo_result.content:
                if hasattr(content, 'text'):
                    print(f"  {content.text}")
            print()
            
            # Test add_numbers tool
            print("Testing add_numbers tool:")
            add_result = await session.call_tool(
                "add_numbers",
                {"a": 15, "b": 27}
            )
            for content in add_result.content:
                if hasattr(content, 'text'):
                    print(f"  {content.text}")
            print()
            
            print("Done!")

if __name__ == "__main__":
    asyncio.run(main())