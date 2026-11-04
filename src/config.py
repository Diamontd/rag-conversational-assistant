"""
Configuration and Client Initialization for RAG Conversational Assistant.
Provides dual-provider OpenAI / Google Gemini API client routing and Langfuse observability.
"""

import os
import json
import time
import logging
import threading
from typing import Dict, Any, List, Optional, Callable
from functools import wraps
from openai import OpenAI

try:
    from langfuse import Langfuse, observe as lf_observe
except ImportError:
    Langfuse = None
    lf_observe = None

logger = logging.getLogger("rag_conversational_assistant.config")


class LocalTraceCollector:
    """Thread-safe local trace collector recording spans to benchmarks/langfuse_traces.json."""
    def __init__(self, trace_file_path: Optional[str] = None):
        if trace_file_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            trace_file_path = os.path.join(base_dir, "benchmarks", "langfuse_traces.json")
        self.trace_file_path = trace_file_path
        self.traces: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def record_trace(self, trace_data: Dict[str, Any]) -> None:
        with self._lock:
            self.traces.append(trace_data)
            self.save()

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.trace_file_path), exist_ok=True)
        with open(self.trace_file_path, "w", encoding="utf-8") as f:
            json.dump(self.traces, f, indent=2)

    def clear(self) -> None:
        with self._lock:
            self.traces = []
            if os.path.exists(self.trace_file_path):
                with open(self.trace_file_path, "w", encoding="utf-8") as f:
                    json.dump([], f)


trace_collector = LocalTraceCollector()


def _discover_gemini_key() -> str:
    """
    Search environment variables and local gitignored .env file for Gemini API Key.
    Strictly avoids hardcoded secrets in source code and respects directory boundaries.
    """
    # 1. Direct environment variables
    for key_name in ["GEMINI_API_KEY", "GOOGLE_API_KEY"]:
        val = os.getenv(key_name)
        if val and val.strip():
            return val.strip()

    # 2. Project-local gitignored .env file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_env = os.path.join(base_dir, ".env")

    if os.path.isfile(project_env):
        try:
            with open(project_env, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY=") or line.startswith("GOOGLE_API_KEY="):
                        parts = line.split("=", 1)
                        if len(parts) == 2 and parts[1].strip():
                            return parts[1].strip().strip('"\'')
        except Exception as e:
            logger.debug(f"Could not read {project_env}: {e}")

    return os.getenv("GEMINI_API_KEY", "")


class AppConfig:
    def __init__(self):
        openai_key = os.getenv("OPENAI_API_KEY")
        gemini_key = _discover_gemini_key()

        if openai_key and not openai_key.startswith("AIza") and not openai_key.startswith("mock"):
            self.provider = "openai"
            self.api_key = openai_key
            self.base_url = None
            self.chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
            default_fallbacks = ["gpt-4o-mini", "gpt-4o"]
            self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        else:
            self.provider = "gemini_compatibility"
            self.api_key = gemini_key or os.getenv("GEMINI_API_KEY", "")
            self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            self.chat_model = os.getenv("CHAT_MODEL", "gemini-2.5-flash-lite")
            default_fallbacks = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]
            self.embedding_model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

        # Ensure chat_model is always the primary model tried, deduplicated with failovers
        self.fallback_models = [self.chat_model] + [m for m in default_fallbacks if m != self.chat_model]

        client_key = self.api_key if self.api_key else "placeholder_key_for_offline_mode"
        if self.base_url:
            self.client = OpenAI(api_key=client_key, base_url=self.base_url)
        else:
            self.client = OpenAI(api_key=client_key)

        # Langfuse Observability Configuration
        self.langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        self.langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        self.langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self.has_remote_langfuse = bool(
            self.langfuse_public_key and self.langfuse_secret_key and Langfuse is not None
        )
        self.langfuse_client = None

        if self.has_remote_langfuse:
            try:
                self.langfuse_client = Langfuse(
                    public_key=self.langfuse_public_key,
                    secret_key=self.langfuse_secret_key,
                    host=self.langfuse_host
                )
                logger.info("Langfuse remote client initialized successfully.")
            except Exception as e:
                logger.warning(f"Could not connect to remote Langfuse: {e}")
                self.has_remote_langfuse = False
        else:
            os.environ["LANGFUSE_RELEASE"] = "disabled"
            logging.getLogger("langfuse").setLevel(logging.ERROR)


config = AppConfig()


def observe(name: Optional[str] = None):
    """
    Tracing decorator recording execution spans.
    Dispatches to Langfuse Cloud (when credentials exist) and LocalTraceCollector.
    """
    def decorator(fn: Callable):
        span_name = name or fn.__name__

        target_fn = fn
        if config.has_remote_langfuse and lf_observe is not None:
            try:
                target_fn = lf_observe(name=span_name)(fn)
            except Exception:
                target_fn = fn

        @wraps(fn)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            error = None
            result = None
            try:
                result = target_fn(*args, **kwargs)
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                latency_ms = round((time.time() - start_time) * 1000, 2)
                span_record = {
                    "span_name": span_name,
                    "timestamp": time.time(),
                    "latency_ms": latency_ms,
                    "status": "ERROR" if error else "SUCCESS",
                    "error": error,
                }
                if hasattr(result, "model_dump"):
                    try:
                        span_record["output"] = result.model_dump(mode="json")
                    except Exception:
                        span_record["output"] = result.model_dump()
                elif hasattr(result, "dict"):
                    span_record["output"] = result.dict()
                elif isinstance(result, (dict, list, str, int, float, bool)):
                    span_record["output_summary"] = str(result)[:300]

                trace_collector.record_trace(span_record)

        return wrapper
    return decorator
