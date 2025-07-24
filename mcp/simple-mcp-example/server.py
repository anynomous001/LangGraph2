#!/usr/bin/env python3
"""
Simple MCP Server Example
Provides basic tools for demonstration
"""

import asyncio
import json
from typing import Dict, Any, List
from mcp.server import Server

import mcp.server.stdio
import mcp.types as types

# Create server instance
server = Server("simple-mcp-server")

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List available tools"""
    return [
        types.Tool(
            name="echo",
            description="Echo back the input text",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to echo back"
                    }
                },
                "required": ["text"]
            }
        ),
        types.Tool(
            name="add_numbers",
            description="Add two numbers together",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "First number"
                    },
                    "b": {
                        "type": "number", 
                        "description": "Second number"
                    }
                },
                "required": ["a", "b"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, 
    arguments: Dict[str, Any]
) -> List[types.TextContent]:
    """Handle tool calls"""
    
    if name == "echo":
        text = arguments.get("text", "")
        return [types.TextContent(
            type="text",
            text=f"Echo: {text}"
        )]
    
    elif name == "add_numbers":
        a = arguments.get("a", 0)
        b = arguments.get("b", 0)
        result = a + b
        return [types.TextContent(
            type="text", 
            text=f"Result: {a} + {b} = {result}"
        )]
    
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    """Run the server"""
    # Run the server using stdio transport
    async with mcp.server.stdio.stdio_server() as streams:
        await server.run(*streams, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())