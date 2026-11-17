"""Optional read-only MCP stdio server; point it at the same SQLite file as the API."""

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from services.evidence import store

server = FastMCP("GeneralDIA evidence")
read_only = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)


@server.tool(annotations=read_only)
def get_evaluation(evaluation_id: str) -> dict:
    """Fetch a stored verdict, checker identity, and measured errors by ID."""
    return store.get_result(store.database_path(), evaluation_id)


@server.tool(annotations=read_only)
def compare_evaluations(left_id: str, right_id: str) -> dict:
    """Compare two stored evaluations; missing measurements are not treated as zero."""
    return store.compare(store.database_path(), left_id, right_id)


if __name__ == "__main__":
    server.run(transport="stdio")
