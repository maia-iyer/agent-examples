"""
OpenTelemetry observability setup for Weather Agent.

Key Features:
- Auto-instrumentation of LangChain with OpenInference
- `create_agent_span` for creating root AGENT spans
- W3C Trace Context propagation for distributed tracing
"""

import logging
import os
from typing import Dict, Any, Optional
from contextlib import contextmanager
from opentelemetry import trace, context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode
from opentelemetry.propagate import set_global_textmap, extract
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.baggage.propagation import W3CBaggagePropagator

logger = logging.getLogger(__name__)

# OpenInference semantic conventions
try:
    from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues
    OPENINFERENCE_AVAILABLE = True
except ImportError:
    OPENINFERENCE_AVAILABLE = False
    logger.warning("openinference-semantic-conventions not available")


def _get_otlp_exporter(endpoint: str):
    """Get HTTP OTLP exporter."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    if not endpoint.endswith("/v1/traces"):
        endpoint = endpoint.rstrip("/") + "/v1/traces"
    return OTLPSpanExporter(endpoint=endpoint)


def setup_observability() -> None:
    """
    Set up OpenTelemetry tracing with OpenInference instrumentation.

    Call this ONCE at agent startup, before importing agent code.
    """
    service_name = os.getenv("OTEL_SERVICE_NAME", "weather-service")
    namespace = os.getenv("K8S_NAMESPACE_NAME", "team1")
    otlp_endpoint = os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "http://otel-collector.kagenti-system.svc.cluster.local:8335"
    )

    logger.info("=" * 60)
    logger.info("Setting up OpenTelemetry observability")
    logger.info(f"  Service: {service_name}")
    logger.info(f"  Namespace: {namespace}")
    logger.info(f"  OTLP Endpoint: {otlp_endpoint}")
    logger.info("=" * 60)

    # Create resource with service attributes
    resource = Resource(attributes={
        "service.name": service_name,
        "service.namespace": namespace,
        "k8s.namespace.name": namespace,
    })

    # Create and configure tracer provider
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(_get_otlp_exporter(otlp_endpoint))
    )
    trace.set_tracer_provider(tracer_provider)

    # Auto-instrument LangChain with OpenInference
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        LangChainInstrumentor().instrument()
        logger.info("LangChain instrumented with OpenInference")
    except ImportError:
        logger.warning("openinference-instrumentation-langchain not available")

    # Configure W3C Trace Context propagation
    set_global_textmap(CompositePropagator([
        TraceContextTextMapPropagator(),
        W3CBaggagePropagator(),
    ]))

    # Instrument OpenAI for GenAI semantic conventions
    try:
        from opentelemetry.instrumentation.openai import OpenAIInstrumentor
        OpenAIInstrumentor().instrument()
        logger.info("OpenAI instrumented with GenAI semantic conventions")
    except ImportError:
        logger.warning("opentelemetry-instrumentation-openai not available")


# Tracer for manual spans - use OpenInference-compatible name
_tracer: Optional[trace.Tracer] = None
TRACER_NAME = "openinference.instrumentation.agent"


def get_tracer() -> trace.Tracer:
    """Get tracer for creating manual spans."""
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer(TRACER_NAME)
    return _tracer


@contextmanager
def create_agent_span(
    name: str = "gen_ai.agent.invoke",
    context_id: Optional[str] = None,
    task_id: Optional[str] = None,
    user_id: Optional[str] = None,
    input_text: Optional[str] = None,
    break_parent_chain: bool = False,
):
    """
    Create an AGENT span with GenAI semantic conventions.

    By default, this preserves the parent chain so the span becomes a child
    of any existing trace context (e.g., A2A request handling). This enables
    full distributed tracing visibility in Phoenix/MLflow.

    Args:
        name: Span name (use gen_ai.agent.* for MLflow AGENT type detection)
        context_id: A2A context_id (becomes gen_ai.conversation.id)
        task_id: A2A task_id
        user_id: User identifier
        input_text: User input message
        break_parent_chain: If True, breaks the trace chain (creates isolated root).
                           Default False to preserve distributed trace visibility.

    Yields:
        The span object (set output.value on it before exiting)
    """
    tracer = get_tracer()

    # Build attributes
    attributes = {}

    # GenAI semantic conventions (for OTEL Collector transforms)
    if context_id:
        attributes["gen_ai.conversation.id"] = context_id
    if input_text:
        attributes["gen_ai.prompt"] = input_text[:1000]
        attributes["input.value"] = input_text[:1000]
    attributes["gen_ai.agent.name"] = "weather-assistant"
    attributes["gen_ai.system"] = "langchain"

    # OpenInference span kind - marks this as an AGENT span
    if OPENINFERENCE_AVAILABLE:
        attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] = OpenInferenceSpanKindValues.AGENT.value

    # Custom attributes for debugging
    if task_id:
        attributes["a2a.task_id"] = task_id
    if user_id:
        attributes["user.id"] = user_id

    # Optional: break the parent chain for isolated traces
    detach_token = None
    if break_parent_chain:
        empty_ctx = context.Context()
        detach_token = context.attach(empty_ctx)

    # Start the span - becomes child of current context (A2A span) by default
    with tracer.start_as_current_span(name, attributes=attributes) as span:
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        finally:
            if detach_token:
                context.detach(detach_token)


@contextmanager
def trace_context_from_headers(headers: Dict[str, str]):
    """
    Activate trace context from HTTP headers.

    Use this to connect to incoming distributed trace.
    """
    ctx = extract(headers)
    token = context.attach(ctx)
    try:
        yield ctx
    finally:
        context.detach(token)
