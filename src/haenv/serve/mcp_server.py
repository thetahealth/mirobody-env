"""Optional MCP stdio access to the exact function-call tools and JSON schemas.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import json


def create_server(session):
    """Import the optional SDK lazily and register the existing ToolSpec objects."""
    try:
        from mcp.server import Server
        from mcp.types import CallToolResult, TextContent, Tool
    except ImportError as exc:
        raise RuntimeError('MCP serving requires the optional extra: pip install "haenv[serve]"') from exc
    server = Server('haenv')

    @server.list_tools()
    async def list_tools():
        return [Tool(name=s.name, description=s.description, inputSchema=s.params) for s in session.specs()]

    # The SDK must not apply stricter argument validation than call_tool does.
    @server.call_tool(validate_input=False)
    async def call(name, arguments):
        result, error = session.call(name, arguments)
        return CallToolResult(content=[TextContent(type='text', text=json.dumps(result, ensure_ascii=False))],
                              isError=bool(error))

    return server


def run(session):
    """Run one session over stdin/stdout until the client disconnects."""
    server = create_server(session)
    import anyio
    from mcp.server.stdio import stdio_server

    async def serve():
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(serve)
