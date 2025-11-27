import asyncio
from urllib.parse import parse_qs, urlparse

from pydantic import AnyUrl

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main():
    """Run the OAuth client example."""
    headers = {"Authorization": "Bearer blablabla"}

    #async with streamablehttp_client("http://localhost:8000/mcp", auth=oauth_auth) as (read, write, _):
    async with streamablehttp_client("http://localhost:8000/mcp", headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print(f"Available tools: {[tool.name for tool in tools.tools]}")

            resources = await session.list_resources()
            print(f"Available resources: {[r.uri for r in resources.resources]}")


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
