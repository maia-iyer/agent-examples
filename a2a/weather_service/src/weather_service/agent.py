import logging
import os
import uvicorn
from textwrap import dedent

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events.event_queue import EventQueue
from starlette.routing import Route
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TaskState, TextPart
from a2a.utils import new_agent_text_message, new_task
from opentelemetry import trace, context as otel_context
from opentelemetry.trace import Link, SpanContext, TraceFlags, NonRecordingSpan
from langchain_core.messages import HumanMessage

from weather_service.graph import get_graph, get_mcpclient

# Get a tracer for adding context_id to spans
tracer = trace.get_tracer(__name__)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# OpenTelemetry GenAI semantic convention instrumentation
# Emits spans with gen_ai.* attributes that get transformed by OTEL Collector
# to OpenInference format (llm.*) for Phoenix and MLflow session metadata
# Emits spans with gen_ai.* attributes that get transformed to OpenInference format
# by the OTEL Collector transform processor before being sent to MLflow
try:
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor
    OpenAIInstrumentor().instrument()
    logger.info("OpenTelemetry GenAI (OpenAI) instrumentation enabled")
except ImportError:
    logger.warning("opentelemetry-instrumentation-openai not available, skipping GenAI instrumentation")


def get_agent_card(host: str, port: int):
    """Returns the Agent Card for the AG2 Agent."""
    capabilities = AgentCapabilities(streaming=True)
    skill = AgentSkill(
        id="weather_assistant",
        name="Weather Assistant",
        description="**Weather Assistant** – Personalized assistant for weather info.",
        tags=["weather"],
        examples=[
            "What is the weather in NY?",
            "What is the weather in Rome?",
        ],
    )
    return AgentCard(
        name="Weather Assistant",
        description=dedent(
            """\
            This agent provides a simple weather information assistance.

            ## Input Parameters
            - **prompt** (string) – the city for which you want to know weather info.

            ## Key Features
            - **MCP Tool Calling** – uses a MCP tool to get weather info.
            """,
        ),
        url=f"http://{host}:{port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=capabilities,
        skills=[skill],
    )

class A2AEvent:
    """
    A class to handle events for A2A Agent.

    Attributes:
        task_updater (TaskUpdater): The task updater instance.
    """

    def __init__(self, task_updater: TaskUpdater):
        self.task_updater = task_updater

    async def emit_event(self, message: str, final: bool = False, failed: bool = False) -> None:
        logger.info("Emitting event %s", message)

        if final or failed:
            parts = [TextPart(text=message)]
            await self.task_updater.add_artifact(parts)
            if final:
                await self.task_updater.complete()
            if failed:
                await self.task_updater.failed()
        else:
            await self.task_updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    message,
                    self.task_updater.context_id,
                    self.task_updater.task_id,
                ),
            )

class WeatherExecutor(AgentExecutor):
    """
    A class to handle weather assistant execution for A2A Agent.
    """
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """
        The agent allows to retrieve weather info through a natural language conversational interface
        """

        # Setup Event Emitter
        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)
        task_updater = TaskUpdater(event_queue, task.id, task.context_id)
        event_emitter = A2AEvent(task_updater)

        # Get user input for the agent
        user_input = context.get_user_input()

        # Parse Messages
        messages = [HumanMessage(content=user_input)]
        input = {"messages": messages}
        logger.info(f'Processing messages: {input}')

        task_updater = TaskUpdater(event_queue, task.id, task.context_id)

        # OPTION A: Break trace chain - create NEW root span for agent
        # This makes gen_ai.agent.invoke the root span so MLflow UI columns work
        # The A2A framework spans will be filtered out by OTEL collector
        #
        # Get parent span context for linking (preserves relationship for debugging)
        parent_span = trace.get_current_span()
        parent_context = parent_span.get_span_context() if parent_span else None
        links = [Link(parent_context)] if parent_context and parent_context.is_valid else []

        # Create a completely clean context with no parent span
        # This forces the tracer to create a new ROOT span with a new trace_id
        # Note: We use an empty Context() and then detach from current context
        clean_context = otel_context.Context()

        # Create a new ROOT span by starting it with a clean context
        # This forces the tracer to create a new trace_id with no parent
        span = tracer.start_span(
            "gen_ai.agent.invoke",
            context=clean_context,  # New trace - no parent
            links=links,  # Keep link to A2A span for debugging
            attributes={
                # GenAI semantic convention attributes
                "gen_ai.conversation.id": task.context_id or "",
                "gen_ai.agent.name": "weather-assistant",
                "gen_ai.agent.id": task.id or "",
                "gen_ai.request.model": "weather-service",
                "gen_ai.system": "langchain",
                # Input message (structured as per GenAI conventions)
                "gen_ai.prompt": user_input[:500] if user_input else "",
                # OpenInference format for Phoenix/MLflow compatibility
                "input.value": user_input[:500] if user_input else "",
                # MLflow UI columns - these are now on ROOT span
                "mlflow.spanInputs": user_input[:500] if user_input else "",
                "mlflow.spanType": "AGENT",
            }
        )

        # Set this span as the current span so child spans are attached to it
        ctx = trace.set_span_in_context(span)
        token = otel_context.attach(ctx)

        try:
            output = None
            # Test MCP connection first
            logger.info(f'Attempting to connect to MCP server at: {os.getenv("MCP_URL", "http://localhost:8000/sse")}')

            mcpclient = get_mcpclient()

            # Try to get tools to verify connection
            try:
                tools = await mcpclient.get_tools()
                logger.info(f'Successfully connected to MCP server. Available tools: {[tool.name for tool in tools]}')
            except Exception as tool_error:
                logger.error(f'Failed to connect to MCP server: {tool_error}')
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(tool_error))
                await event_emitter.emit_event(f"Error: Cannot connect to MCP weather service at {os.getenv('MCP_URL', 'http://localhost:8000/sse')}. Please ensure the weather MCP server is running. Error: {tool_error}", failed=True)
                return

            graph = await get_graph(mcpclient)
            async for event in graph.astream(input, stream_mode="updates"):
                await event_emitter.emit_event(
                    "\n".join(
                        f"🚶‍♂️{key}: {str(value)[:256] + '...' if len(str(value)) > 256 else str(value)}"
                        for key, value in event.items()
                    )
                    + "\n"
                )
                output = event
                logger.info(f'event: {event}')
            output = output.get("assistant", {}).get("final_answer")
            # Add response to span using GenAI conventions
            # Now that this span IS the root span, MLflow UI will read from it
            if output:
                span.set_attribute("gen_ai.completion", str(output)[:500])
                # OpenInference format for Phoenix/MLflow compatibility
                span.set_attribute("output.value", str(output)[:500])
                # MLflow UI columns - Response (now on ROOT span)
                span.set_attribute("mlflow.spanOutputs", str(output)[:500])
            await event_emitter.emit_event(str(output), final=True)
        except Exception as e:
            logger.error(f'Graph execution error: {e}')
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            await event_emitter.emit_event(f"Error: Failed to process weather request. {str(e)}", failed=True)
            raise Exception(str(e))
        finally:
            # Always end the span and detach the context
            span.end()
            otel_context.detach(token)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Not implemented
        """
        raise Exception("cancel not supported")

def run():
    """
    Runs the A2A Agent application.
    """
    agent_card = get_agent_card(host="0.0.0.0", port=8000)

    request_handler = DefaultRequestHandler(
        agent_executor=WeatherExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    # Build the Starlette app
    app = server.build()

    # Add the new agent-card.json path alongside the legacy agent.json path
    app.routes.insert(0, Route(
        '/.well-known/agent-card.json',
        server._handle_get_agent_card,
        methods=['GET'],
        name='agent_card_new',
    ))

    # Add middleware to log all incoming requests with headers
    
    @app.middleware("http")
    async def log_authorization_header(request, call_next):
        auth_header = request.headers.get("authorization", "No Authorization header")
        logger.info(f"🔐 Incoming request to {request.url.path} with Authorization: {auth_header[:80] + '...' if len(auth_header) > 80 else auth_header}")
        response = await call_next(request)
        return response

    uvicorn.run(app, host="0.0.0.0", port=8000)
