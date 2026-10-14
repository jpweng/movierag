"""Optional: see every step of every query in a UI (Arize Phoenix, OpenTelemetry).

Install the extra:   uv sync --extra tracing
Start the UI:        uv run phoenix serve          (then open http://localhost:6006)
In the notebook:     from movierag.tracing import enable_tracing; enable_tracing()

LlamaIndex emits events for every retrieve / rerank / LLM call. OpenInference
turns them into OpenTelemetry spans, and Phoenix stores and displays them:
inputs, outputs, latency and token counts per step.
"""

from __future__ import annotations


def enable_tracing(project: str = "movierag", endpoint: str = "http://localhost:6006/v1/traces") -> bool:
    try:
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
        from phoenix.otel import register
    except ImportError:
        print("Tracing extras missing. Run: uv sync --extra tracing")
        return False
    tracer_provider = register(project_name=project, endpoint=endpoint)
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    print(f"Tracing on -> {endpoint.rsplit('/v1', 1)[0]} (project '{project}')")
    return True
