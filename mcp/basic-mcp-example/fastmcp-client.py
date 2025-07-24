#!/usr/bin/env python3
"""
Advanced MCP Client for FastMCP Server
Tests web search, PDF reader, calculator, and other advanced tools
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_web_search(session):
    """Test the web search functionality"""
    print("🔍 Testing Web Search:")
    print("-" * 40)
    
    try:
        result = await session.call_tool(
            "web_search",
            {"query": "Python programming language", "max_results": 3}
        )
        for content in result.content:
            if hasattr(content, 'text'):
                print(content.text)
    except Exception as e:
        print(f"Web search error: {e}")
    print("\n")

async def test_calculator(session):
    """Test the calculator functionality"""
    print("🧮 Testing Calculator:")
    print("-" * 40)
    
    test_expressions = [
        "2 + 3 * 4",
        "sqrt(16) + pow(2, 3)",
        "sin(pi/2) + cos(0)",
        "factorial(5)",
        "log(100) + ln(e)"
    ]
    
    for expr in test_expressions:
        try:
            result = await session.call_tool("calculator", {"expression": expr})
            for content in result.content:
                if hasattr(content, 'text'):
                    print(content.text)
            print()
        except Exception as e:
            print(f"Calculator error for '{expr}': {e}\n")

async def test_file_operations(session):
    """Test file operations"""
    print("📁 Testing File Operations:")
    print("-" * 40)
    
    # Create a test file
    try:
        result = await session.call_tool(
            "file_operations",
            {
                "operation": "create",
                "file_path": "test_file.txt",
                "content": "Hello, this is a test file created by MCP!\nLine 2 of the test file.\nLine 3 with some numbers: 123 456."
            }
        )
        for content in result.content:
            if hasattr(content, 'text'):
                print(content.text)
        print()
        
        # Read the file back
        result = await session.call_tool(
            "file_operations",
            {"operation": "read", "file_path": "test_file.txt"}
        )
        for content in result.content:
            if hasattr(content, 'text'):
                print(content.text)
        print()
        
        # Get file info
        result = await session.call_tool(
            "file_operations",
            {"operation": "info", "file_path": "test_file.txt"}
        )
        for content in result.content:
            if hasattr(content, 'text'):
                print(content.text)
        print()
        
        # Clean up - delete the test file
        result = await session.call_tool(
            "file_operations",
            {"operation": "delete", "file_path": "test_file.txt"}
        )
        for content in result.content:
            if hasattr(content, 'text'):
                print(content.text)
        print()
        
    except Exception as e:
        print(f"File operations error: {e}\n")

async def test_text_analyzer(session):
    """Test text analysis functionality"""
    print("📝 Testing Text Analyzer:")
    print("-" * 40)
    
    sample_text = """
    The quick brown fox jumps over the lazy dog. This is a sample text for analysis.
    It contains multiple sentences and various words. Some words repeat, like 'the' and 'is'.
    This text will help demonstrate the text analysis capabilities of our MCP server.
    """
    
    try:
        result = await session.call_tool(
            "text_analyzer",
            {"text": sample_text, "analysis_type": "all"}
        )
        for content in result.content:
            if hasattr(content, 'text'):
                print(content.text)
        print()
    except Exception as e:
        print(f"Text analyzer error: {e}\n")

async def test_hash_generator(session):
    """Test hash generation functionality"""
    print("🔐 Testing Hash Generator:")
    print("-" * 40)
    
    test_text = "Hello, World! This is a test string for hashing."
    
    try:
        result = await session.call_tool(
            "hash_generator",
            {"text": test_text, "hash_type": "all"}
        )
        for content in result.content:
            if hasattr(content, 'text'):
                print(content.text)
        print()
    except Exception as e:
        print(f"Hash generator error: {e}\n")

async def test_pdf_reader(session):
    """Test PDF reading functionality"""
    print("📄 Testing PDF Reader:")
    print("-" * 40)
    
    # Note: This will only work if you have a PDF file to test with
    try:
        result = await session.call_tool(
            "read_pdf",
            {"file_path": "sample.pdf", "page_range": "1"}
        )
        for content in result.content:
            if hasattr(content, 'text'):
                print(content.text)
        print()
    except Exception as e:
        print(f"PDF reader test skipped (no sample PDF): {e}\n")

async def main():
    """Main client function"""
    
    # Server parameters
    server_params = StdioServerParameters(
        command="python",
        args=["fastmcp_server.py"]  # Adjust path as needed
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                
                # Initialize the session
                await session.initialize()
                
                print("🚀 Connected to Advanced MCP Server!")
                print("=" * 60)
                
                # List available tools
                tools_result = await session.list_tools()
                print(f"📋 Available tools: {len(tools_result.tools)}")
                for tool in tools_result.tools:
                    print(f"   • {tool.name}: {tool.description}")
                print("\n")
                
                # Test all the tools
                await test_calculator(session)
                await test_file_operations(session)
                await test_text_analyzer(session)
                await test_hash_generator(session)
                # await test_web_search(session)
                await test_pdf_reader(session)
                
                print("✅ All tests completed!")
                
    except Exception as e:
        print(f"❌ Client error: {e}")
        print("\nMake sure you have:")
        print("1. Installed required packages: pip install fastmcp aiohttp PyPDF2")
        print("2. The server file is named 'fastmcp_server.py' in the same directory")
        print("3. Python can execute the server script")

if __name__ == "__main__":
    asyncio.run(main())