"""MCP Server for dataflow analysis tools."""
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .tools import trace_symbol, trace_callchain

server = Server("think-more-dataflow")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="trace_symbol",
            description="Trace a symbol's definitions and references in the codebase",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The symbol name to trace",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to the source file",
                    },
                    "project_path": {
                        "type": "string",
                        "description": "Root path of the project",
                    },
                },
                "required": ["symbol", "file_path", "project_path"],
            },
        ),
        Tool(
            name="trace_callchain",
            description="Trace the call chain of a function",
            inputSchema={
                "type": "object",
                "properties": {
                    "entry_point": {
                        "type": "string",
                        "description": "The function name to trace",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to the source file",
                    },
                    "project_path": {
                        "type": "string",
                        "description": "Root path of the project",
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["callers", "callees"],
                        "description": "Direction to trace",
                        "default": "callers",
                    },
                },
                "required": ["entry_point", "file_path", "project_path"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a tool."""
    import json

    if name == "trace_symbol":
        result = trace_symbol(
            symbol=arguments["symbol"],
            file_path=arguments["file_path"],
            project_path=arguments["project_path"],
        )
    elif name == "trace_callchain":
        result = trace_callchain(
            entry_point=arguments["entry_point"],
            file_path=arguments["file_path"],
            project_path=arguments["project_path"],
            direction=arguments.get("direction", "callers"),
        )
    else:
        result = {"error": f"Unknown tool: {name}"}

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


def run():
    """Entry point for the MCP server."""
    import asyncio
    asyncio.run(main())


if __name__ == "__main__":
    run()
