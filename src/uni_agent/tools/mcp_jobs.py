import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

_client = None


def get_mcp_jobs_client() -> MultiServerMCPClient:
    """Get or create the MCP jobs client."""
    global _client
    if _client is None:
        _client = MultiServerMCPClient(
            {
                "mcp_jobs": {
                    "transport": "http",
                    "url": "http://192.168.0.201:6000/mcp",
                }
            }
        )
    return _client


async def get_mcp_tools():
    """Get tools from MCP client with proper initialization."""
    try:
        client = get_mcp_jobs_client()
        
        # Ensure client is initialized if needed
        if hasattr(client, 'initializeConnections'):
            try:
                # Add timeout to prevent hanging
                await asyncio.wait_for(client.initializeConnections(), timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                # If initialization fails, continue anyway
                pass
        
        # Get tools - this should return a list
        # Add timeout to prevent hanging
        tools = await asyncio.wait_for(client.get_tools(), timeout=10.0)
        
        # Ensure we return a list
        if isinstance(tools, list):
            return tools
        elif hasattr(tools, '__iter__') and not isinstance(tools, (str, bytes)):
            return list(tools)
        else:
            # If tools is not a list or iterable, return empty list
            return []
    except asyncio.TimeoutError:
        # Timeout - server may be unavailable
        import logging
        logging.warning("Timeout getting MCP tools - server may be unavailable")
        return []
    except Exception as e:
        # Log error and return empty list
        import logging
        logging.warning(f"Error getting MCP tools: {e}")
        return []
