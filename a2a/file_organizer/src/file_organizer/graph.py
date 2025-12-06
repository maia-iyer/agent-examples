import os
from langgraph.graph import StateGraph, MessagesState, START
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_openai import ChatOpenAI

from file_organizer.configuration import Configuration

config = Configuration()

# Extend MessagesState to include a final answer
class ExtendedMessagesState(MessagesState):
    final_answer: str = ""

def get_mcpclient():
    return MultiServerMCPClient({
        "cloud_storage": {
            "url": os.getenv("MCP_URL", "http://cloud-storage-tool:8000/mcp"),
            "transport": os.getenv("MCP_TRANSPORT", "streamable_http"),
        }
    })

async def get_graph(client) -> StateGraph:
    llm = ChatOpenAI(
        model=config.llm_model,
        openai_api_key=config.llm_api_key,
        openai_api_base=config.llm_api_base,
        temperature=0,
    )

    # Get tools asynchronously
    tools = await client.get_tools()
    llm_with_tools = llm.bind_tools(tools)
    bucket_uri = os.getenv("BUCKET_URI")

    bucket_info = f"Target bucket: {bucket_uri}" if bucket_uri else "No bucket URI configured. Ask the user to specify which bucket to organize."

    sys_msg = SystemMessage(content=(
    "You are a file-organization agent that must use tools faithfully.\n\n"
    "The following are the authoritative tool schemas you MUST obey exactly:\n\n"
    f"{bucket_info}\n"
    "TOOL: get_objects\n"
    "INPUT:\n"
    "  {\n"
    "    \"bucket_uri\": \"string\"\n"
    "  }\n\n"
    "TOOL: perform_action\n"
    "INPUT:\n"
    "  {\n"
    "    \"file_uri\": \"string\",\n"
    "    \"action\": \"move\" | \"copy\",\n"
    "    \"target_uri\": \"string\"\n"
    "  }\n\n"
    "RULES:\n"
    "1. You MUST NOT include any fields other than those defined above.\n"
    "   Forbidden fields include: source_uri, target_folder_uri, status,\n"
    "   filename, message, or any other keys not listed in the schema.\n\n"
    "2. When organizing files:\n"
    "   - ALWAYS begin by calling get_objects.\n"
    "   - Then produce one perform_action call per file that needs moving.\n"
    "   - target_uri must represent a folder and end with '/'.\n"
    "   - file_uri must be the exact file_uri from get_objects output.\n\n"
    "3. NEVER wrap tool calls in narrative text.\n"
    "   A tool call MUST be the only content in the assistant's message.\n\n"
    "4. NEVER ask the user questions unless the user explicitly requests an explanation.\n\n"
    "5. NEVER summarize files before or after tools. Tool calls must happen immediately.\n\n"
    "6. After all perform_action calls have run, you may provide a brief summary as plain text.\n"
))


    # Node
    def assistant(state: ExtendedMessagesState) -> ExtendedMessagesState:
        result = llm_with_tools.invoke([sys_msg] + state["messages"])
        state["messages"].append(result)
        # Set the final answer only if the result is an AIMessage (i.e., not a tool call)
        # and it's meant to be the final response to the user.
        # This logic might need refinement based on when you truly consider the answer "final".
        if isinstance(result, AIMessage) and not result.tool_calls:
            state["final_answer"] = result.content
        return state

    # Build graph
    builder = StateGraph(ExtendedMessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # Compile graph
    graph = builder.compile()
    return graph
