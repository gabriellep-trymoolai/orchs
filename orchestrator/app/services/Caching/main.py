# main.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import io, hashlib, time

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from time import time as now

try:
    from .settings import settings
    from .cache import RedisCache, PromptRequest, CacheWarmRequest, process_prompt, embed_text, store_response
except ImportError:
    # Fallback for when imported from outside the package
    from settings import settings
    from cache import RedisCache, PromptRequest, CacheWarmRequest, process_prompt, embed_text, store_response

# -------- Prometheus HTTP Metrics --------
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests",
    ["method", "endpoint", "http_status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency (seconds)",
    ["method", "endpoint"]
)

# Middleware for metrics
class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = now()
        response = await call_next(request)
        process_time = now() - start_time
        endpoint = request.url.path
        REQUEST_COUNT.labels(request.method, endpoint, response.status_code).inc()
        REQUEST_LATENCY.labels(request.method, endpoint).observe(process_time)
        return response

app = FastAPI(title="CacheBot API", version="1.1")
app.add_middleware(MetricsMiddleware)

cache = RedisCache()
# Load persisted runtime config - on startup so values survive restarts
cache.load_config()

start_time = time.time()

# ---------------- Metrics ----------------
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ------------- Config (GET/POST, persisted) -------------
def _current_config():
    return {
        "enabled": settings.ENABLED,
        "ttl": settings.CACHE_TTL,
        "similarity_threshold": settings.SIMILARITY_THRESHOLD,
    }

@app.get("/cache/config")
def get_cache_config():
    return _current_config()

class ConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    ttl: Optional[int] = None
    similarity_threshold: Optional[float] = None

@app.post("/cache/config")
def update_cache_config(payload: ConfigUpdate):
    # Use cache setters so values are validated and persisted in Redis
    if payload.enabled is not None:
        cache.set_enabled(payload.enabled)
    if payload.ttl is not None:
        try:
            cache.set_ttl(payload.ttl)
        except ValueError as e:
            raise HTTPException(400, str(e))
    if payload.similarity_threshold is not None:
        try:
            cache.set_similarity_threshold(payload.similarity_threshold)
        except ValueError as e:
            raise HTTPException(400, str(e))
    return {"message": "Cache config updated", "config": _current_config()}

# ---------------- Cache listing (paginated) ----------------
@app.get("/cache/keys")
def list_keys(page: int = 1, page_size: int = 100):
    """
    Paginated listing to avoid huge payloads.
    Use ?page=1&page_size=100 (max page_size=1000)
    """
    page = max(1, page)
    page_size = max(1, min(page_size, 1000))
    all_keys = cache.list_keys()
    total = len(all_keys)
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "keys": all_keys[start:end],
    }

@app.delete("/cache/keys")
def clear_cache():
    cache.clear()
    return {"message": "Cache cleared"}

@app.get("/cache/stats")
def cache_stats():
    return cache.stats()

# --- legacy exports (key/value) ---
@app.get("/cache/export/csv")
def export_cache_csv():
    content = cache.export_csv()
    return StreamingResponse(
        io.StringIO(content),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cache.csv"},
    )

@app.get("/cache/export/json")
def export_cache_json():
    return cache.export_json()

# --- rich exports (session_id + prompt/response + metadata + source) ---
@app.get("/cache/export/full")
def export_cache_full_json():
    return cache.export_records()

@app.get("/cache/export/full.csv")
def export_cache_full_csv():
    content = cache.export_full_csv()
    return StreamingResponse(
        io.StringIO(content),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cache_full.csv"},
    )

# ---------------- Health ----------------
@app.get("/health")
def health_check():
    redis_status = "connected" if cache.ping() else "disconnected"
    uptime_seconds = int(time.time() - start_time)
    return {
        "status": "ok" if redis_status == "connected" else "degraded",
        "redis": redis_status,
        "uptime_seconds": uptime_seconds,
        "key_count": len(cache.list_keys()),
        "cache_enabled": settings.ENABLED,
    }

# ---------------- Warm cache ----------------
@app.post("/cache/warm")
def warm_cache(data: CacheWarmRequest):
    """
    - embed_only: writes vector and a base payload with source='warmed_embed_only'
    - full: generates response and writes base payload with source='warmed'
    """
    full_mode = data.mode == "full"
    for prompt in data.prompts:
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        base_key = f"chat:{data.session_id}:{prompt_hash}"

        embedding = embed_text(prompt)
        if embedding is not None:
            cache.set(f"{base_key}:vec", embedding.tolist())

        if full_mode:
            # computing fresh response once, but marking the stored entry as warmed
            resp_text = f"Answer to: {prompt}"
            label = "General.Uncategorized"
            cache.set(base_key, {"prompt": prompt, "response": resp_text, "label": label, "source": "warmed"})
        else:
            cache.set(base_key, {"prompt": prompt, "response": "<WARMED>", "label": None, "source": "warmed_embed_only"})

    return {"message": f"Cache warmed with {len(data.prompts)} prompts", "mode": data.mode}

# ---------------- Process prompt ----------------
@app.post("/process_prompt")
def process_prompt_endpoint(req: PromptRequest):
    final_message = req.message or req.prompt
    if not final_message:
        raise HTTPException(400, "Either 'message' or 'prompt' must be provided")
    req.message = final_message
    return process_prompt(cache, req)

class StoreRequest(BaseModel):
    session_id: str
    message: str
    response: str

@app.post("/store_response")
def store_response_endpoint(req: StoreRequest):
    """Store a fresh response in the cache"""
    store_response(cache, req.session_id, req.message, req.response)
    return {"message": "Response stored in cache", "session_id": req.session_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
