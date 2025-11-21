"""Activity logging middleware and storage for API requests."""

import time
from collections import deque
from datetime import datetime
from typing import Deque, Dict, Any, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import json


class ActivityLog:
    """Store for activity logs with fixed size."""

    def __init__(self, max_size: int = 200):
        self._logs: Deque[Dict[str, Any]] = deque(maxlen=max_size)

    def add_log(self, log_entry: Dict[str, Any]) -> None:
        """Add a log entry to the store."""
        self._logs.append(log_entry)

    def get_logs(self, limit: Optional[int] = None) -> list[Dict[str, Any]]:
        """Get recent logs, optionally limited."""
        logs = list(self._logs)
        logs.reverse()  # Most recent first
        if limit:
            return logs[:limit]
        return logs

    def clear_logs(self) -> None:
        """Clear all logs."""
        self._logs.clear()


# Global activity log storage
activity_log = ActivityLog(max_size=200)


class ActivityLoggerMiddleware(BaseHTTPMiddleware):
    """Middleware to log all API requests and responses."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Start timing
        start_time = time.time()

        # Extract request details
        method = request.method
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"

        # Extract model from request body if present
        model_name = None
        request_preview = None

        # Only try to parse body for POST requests to specific endpoints
        if method == "POST" and path in ["/v1/chat/completions", "/v1/embeddings"]:
            try:
                # Read body
                body = await request.body()
                is_streaming = False

                if body:
                    body_json = json.loads(body.decode())
                    model_name = body_json.get("model")
                    is_streaming = body_json.get("stream", False)

                    # Create sanitized preview (hide images, truncate long text)
                    request_preview = {}
                    if "messages" in body_json:
                        # Chat request
                        request_preview["messages"] = f"{len(body_json['messages'])} messages"
                        if body_json["messages"]:
                            last_msg = body_json["messages"][-1]
                            if isinstance(last_msg.get("content"), str):
                                content = last_msg["content"]
                                request_preview["last_message"] = content[:50] + "..." if len(content) > 50 else content
                    elif "input" in body_json:
                        # Embeddings request
                        input_data = body_json["input"]
                        if isinstance(input_data, list):
                            request_preview["input"] = f"{len(input_data)} texts"
                        else:
                            request_preview["input"] = input_data[:50] + "..." if len(input_data) > 50 else input_data

                # Restore body for the actual request handler (but not for streaming requests)
                if not is_streaming:
                    async def receive():
                        return {"type": "http.request", "body": body}

                    request._receive = receive
            except Exception:
                # If we can't parse body, continue anyway
                pass

        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
            error = None
        except Exception as e:
            status_code = 500
            error = str(e)
            raise
        finally:
            # Calculate response time
            response_time_ms = int((time.time() - start_time) * 1000)

            # Create log entry
            log_entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "method": method,
                "path": path,
                "model": model_name,
                "status_code": status_code,
                "response_time_ms": response_time_ms,
                "client_ip": client_ip,
                "request_preview": request_preview,
                "error": error,
            }

            # Skip logging certain paths to avoid clutter
            skip_paths = ["/system/stats", "/api/activity-logs", "/health"]
            if path not in skip_paths and not path.startswith("/static"):
                activity_log.add_log(log_entry)

        return response
