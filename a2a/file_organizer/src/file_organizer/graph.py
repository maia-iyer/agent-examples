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

    # Get tools and bind them (This handles the Schema!)
    tools = await client.get_tools()
    llm_with_tools = llm.bind_tools(tools)
    
    bucket_uri = os.getenv("BUCKET_URI")
    bucket_context = f"Target bucket: {bucket_uri}" if bucket_uri else ""

    # CLEANER SYSTEM PROMPT
    sys_msg = SystemMessage(content=dedent(f"""
    You are a precise file-organization assistant.
    {bucket_context}

    ## Standard Operating Procedure
    1. **ANALYSIS:** Check if the user wants to 'move' or 'copy' files.
    2. **DISCOVERY:** You MUST run `get_objects` first to identify available files and their specific URIs.
    3. **EXECUTION:** Call `perform_action` for each file you need to organize.
       - Use the exact URI returned by `get_objects`.
       - Ensure `target_uri` represents a folder and ends with a trailing slash '/'.

    ## Critical Rules
    - Do not invent filenames. Only use files found in the DISCOVERY step.
    - If the user did not specify a bucket, ask them for it.
    - Keep your final response brief.
    """))

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
