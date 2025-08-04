import logging
import os
from textwrap import dedent
from typing import AsyncIterator

from acp_sdk import Metadata, Message, Link, LinkType, MessagePart
from acp_sdk.models.errors import ACPError, Error, ErrorCode
from acp_sdk.server import Server
from openinference.instrumentation.langchain import LangChainInstrumentor
from pydantic import AnyUrl
from langchain_core.messages import HumanMessage

from acp_slack_assistant.graph import get_graph, get_mcpclient

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

LangChainInstrumentor().instrument()

server = Server()

@server.agent(
    metadata=Metadata(
        programming_language="Python",
        license="Apache 2.0",
        framework="LangGraph",
        links=[
            Link(
                type=LinkType.SOURCE_CODE,
                url=AnyUrl(
                    f"https://github.com/i-am-bee/beeai-platform/blob/{os.getenv('RELEASE_VERSION', 'main')}"
                    "/agents/community/acp-slack-assistant"
                ),
            )
        ],
        documentation=dedent(
            """\
            This agent provides a simple slack assistance.

            ## Input Parameters
            - **prompt** (string) – ask a question about the content of your slack.

            ## Key Features
            - **MCP Tool Calling** – uses a MCP tool to get slack info.
            """,
        ),
        use_cases=[
            "**Slack Assistant** – Personalized assistant for slack info.",
        ],
        env=[
            {"name": "LLM_MODEL", "description": "Model to use from the specified OpenAI-compatible API."},
            {"name": "LLM_API_BASE", "description": "Base URL for OpenAI-compatible API endpoint"},
            {"name": "LLM_API_KEY", "description": "API key for OpenAI-compatible API endpoint"},
            {"name": "MCP_URL", "description": "MCP Server URL for the slack tool"},
            {"name": "ACP_MCP_TRANSPORT", "description": "MCP transport type: sse, stdio, streamable_http, websocket (defaults to 'streamable_http')"},
        ],
        ui={"type": "hands-off", "user_greeting": "Ask me about the general slack"},
        examples={
            "cli": [
                {
                    "command": 'beeai run acp_slack_assistant "what channels are in the slack space?"',
                    "description": "Running a Slack Query",
                    "processing_steps": [
                        "Calls the slack MCP tool to get the slack info"
                        "Parses results and return it",
                    ],
                }
            ]
        },
    )
)
async def acp_slack_assistant(input: list[Message]) -> AsyncIterator:
    """
    The agent allows to retrieve slack info through a natural language conversational interface
    """
    messages = [HumanMessage(content=input[-1].parts[-1].content)]
    input = {"messages": messages}
    logger.info(f'Processing messages: {input}')

    try:
        output = None
        # Test MCP connection first
        logger.info(f'Attempting to connect to MCP server at: {os.getenv("MCP_URL", "http://localhost:8000/mcp")}')
        
        mcpclient = get_mcpclient()
        
        # Try to get tools to verify connection
        try:
            tools = await mcpclient.get_tools()
            logger.info(f'Successfully connected to MCP server. Available tools: {[tool.name for tool in tools]}')
        except Exception as tool_error:
            logger.error(f'Failed to connect to MCP server: {tool_error}')
            yield MessagePart(content=f"Error: Cannot connect to MCP slack service at {os.getenv('MCP_URL', 'http://localhost:8000/mcp')}. Please ensure the slack MCP server is running. Error: {tool_error}")
            return
            
        graph = await get_graph(mcpclient)
        async for event in graph.astream(input, stream_mode="updates"):
            yield {
                "message": "\n".join(
                    f"🚶‍♂️{key}: {str(value)[:100] + '...' if len(str(value)) > 100 else str(value)}"
                    for key, value in event.items()
                )
                + "\n"
            }
            output = event
            logger.info(f'event: {event}')
        output =  output.get("assistant", {}).get("final_answer")
        yield MessagePart(content=str(output))
    except Exception as e:
        logger.error(f'Graph execution error: {e}')
        yield MessagePart(content=f"Error: Failed to process slack request. {str(e)}")
        raise ACPError(Error(code=ErrorCode.SERVER_ERROR, message=str(e)))


def run():
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))


if __name__ == "__main__":
    run()
