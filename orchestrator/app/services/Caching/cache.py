# cache.py
from __future__ import annotations

import csv
import hashlib
import io
import json
import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import redis
import zlib
from pydantic import BaseModel, Field, model_validator
from prometheus_client import Counter
from sentence_transformers import SentenceTransformer

try:
    from .settings import settings
except ImportError:
    # Fallback for when imported from outside the package
    from settings import settings

# --------- Prometheus Metrics ---------
CACHE_HITS = Counter("cache_hits_total", "Total cache hits", ["type"])
CACHE_MISSES = Counter("cache_misses_total", "Total cache misses")

# --------- Config persistence ---------
CONFIG_KEY = "chat:v1:config"

def _coerce_bool(v: Any, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "t", "yes", "y"}
    return default

def _coerce_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default

def _coerce_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


# --------- Pydantic Models ---------
class PromptRequest(BaseModel):
    session_id: str = Field(..., description="Unique session ID")
    message: Optional[str] = None
    prompt: Optional[str] = None

    @model_validator(mode="after")
    def _require_message_or_prompt(self) -> "PromptRequest":
        if not (self.message or self.prompt):
            raise ValueError("Either 'message' or 'prompt' must be provided")
        return self


class PromptResponse(BaseModel):
    session_id: str
    response: str = ""
    from_cache: bool
    similarity: Optional[float] = None
    label: Optional[str] = None


class CacheWarmRequest(BaseModel):
    session_id: str
    prompts: List[str]
    mode: str = Field("embed_only", description="'embed_only' or 'full' mode")


# --------- Redis Cache ---------
class RedisCache:
    """Redis backend with compression, metadata, persistence, and utilities."""
    def __init__(self):
        self.client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=False,
            socket_timeout=2,
            socket_connect_timeout=2,
            health_check_interval=30,
        )

    # ---- Config persistence ----
    def load_config(self) -> None:
        """Load persisted runtime config from Redis into settings."""
        raw = self.get(CONFIG_KEY)
        if not isinstance(raw, dict):
            return
        settings.ENABLED = _coerce_bool(raw.get("ENABLED"), settings.ENABLED)
        settings.CACHE_TTL = _coerce_int(raw.get("CACHE_TTL"), settings.CACHE_TTL)
        settings.SIMILARITY_THRESHOLD = _coerce_float(
            raw.get("SIMILARITY_THRESHOLD"), settings.SIMILARITY_THRESHOLD
        )

    def save_config(self) -> None:
        """Persist current settings to Redis - non-expiring)."""
        try:
            key = self._versioned(CONFIG_KEY)
            payload = {
                "ENABLED": settings.ENABLED,
                "CACHE_TTL": settings.CACHE_TTL,
                "SIMILARITY_THRESHOLD": settings.SIMILARITY_THRESHOLD,
            }
            self.client.set(key, self._compress(payload))
        except redis.exceptions.RedisError as e:
            print(f"[RedisCache] SAVE_CONFIG failed: {e}")

    # programmatic setters (agent-friendly)
    def set_enabled(self, enabled: bool) -> None:
        settings.ENABLED = bool(enabled)
        self.save_config()

    def set_ttl(self, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            raise ValueError("TTL must be positive")
        settings.CACHE_TTL = int(ttl_seconds)
        self.save_config()

    def set_similarity_threshold(self, value: float) -> None:
        if not (0.0 <= value <= 1.0):
            raise ValueError("Similarity threshold must be between 0 and 1")
        settings.SIMILARITY_THRESHOLD = float(value)
        self.save_config()

    # ---- Core Redis helpers ----
    def ping(self) -> bool:
        try:
            return self.client.ping()
        except redis.exceptions.RedisError:
            return False

    def _versioned(self, key: str) -> str:
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        if key.startswith("chat:v1:"):
            return key
        if key.startswith("chat:"):
            return f"chat:v1:{key[len('chat:'):]}"
        return key

    def _compress(self, value: Any) -> bytes:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, dict)):
            value = json.dumps(value)
        return zlib.compress(value.encode())

    def _decompress(self, value: bytes) -> Any:
        decompressed = zlib.decompress(value).decode()
        try:
            return json.loads(decompressed)
        except json.JSONDecodeError:
            return decompressed

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if not settings.ENABLED:
            return
        ttl = ttl or settings.CACHE_TTL
        now = time.time()
        key = self._versioned(key)
        try:
            self.client.setex(key, ttl, self._compress(value))
            meta = {"created_at": now, "last_accessed": now, "model": "all-MiniLM-L6-v2"}
            self.client.setex(f"{key}:meta", ttl, self._compress(meta))
        except redis.exceptions.RedisError as e:
            print(f"[RedisCache] SET failed: {e}")

    def get(self, key: str) -> Optional[Any]:
        key = self._versioned(key)
        try:
            val = self.client.get(key)
            if val:
                meta_key = f"{key}:meta"
                if self.client.exists(meta_key):
                    meta = self._decompress(self.client.get(meta_key))
                    if isinstance(meta, dict):
                        meta["last_accessed"] = time.time()
                        self.client.setex(meta_key, settings.CACHE_TTL, self._compress(meta))
                return self._decompress(val)
        except redis.exceptions.RedisError as e:
            print(f"[RedisCache] GET failed: {e}")
        return None

    def scan_iter(self, match: Optional[str] = None) -> Iterable[bytes]:
        try:
            return self.client.scan_iter(match=match)
        except redis.exceptions.RedisError as e:
            print(f"[RedisCache] SCAN failed: {e}")
            return []

    def list_keys(self) -> List[Dict[str, Any]]:
        """Return base keys and their metadata (skip :meta)."""
        try:
            keys = [k.decode() for k in self.client.scan_iter(match="chat:v1:*") if b":meta" not in k]
            result = []
            for k in keys:
                meta_raw = self.client.get(f"{k}:meta")
                meta = self._decompress(meta_raw) if meta_raw else {}
                result.append({"key": k, **(meta if isinstance(meta, dict) else {})})
            return result
        except redis.exceptions.RedisError as e:
            print(f"[RedisCache] LIST_KEYS failed: {e}")
            return []

    def clear(self) -> None:
        try:
            self.client.flushdb()
        except redis.exceptions.RedisError as e:
            print(f"[RedisCache] CLEAR failed: {e}")

    # --- simple (legacy) exports ---
    def export_json(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        try:
            for k in self.client.scan_iter(match="chat:v1:*"):
                if (isinstance(k, bytes) and k.endswith(b":meta")) or (isinstance(k, str) and k.endswith(":meta")):
                    continue
                k_str = k.decode() if isinstance(k, bytes) else str(k)
                data[k_str] = self.get(k_str)
        except redis.exceptions.RedisError as e:
            print(f"[RedisCache] EXPORT_JSON failed: {e}")
        return data

    def export_csv(self) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Key", "Value"])
        try:
            for k in self.client.scan_iter(match="chat:v1:*"):
                if (isinstance(k, bytes) and k.endswith(b":meta")) or (isinstance(k, str) and k.endswith(":meta")):
                    continue
                k_str = k.decode() if isinstance(k, bytes) else str(k)
                writer.writerow([k_str, json.dumps(self.get(k_str))])
        except redis.exceptions.RedisError as e:
            print(f"[RedisCache] EXPORT_CSV failed: {e}")
        return output.getvalue()

    # --- rich export (session_id + prompt/response + metadata + source) ---
    def _parse_key(self, k: str) -> tuple[Optional[str], Optional[str], bool]:
        s = k.decode() if isinstance(k, bytes) else str(k)
        s = self._versioned(s)
        is_vec = s.endswith(":vec")
        if is_vec:
            s_base = s[:-4]
        else:
            s_base = s
        parts = s_base.split(":")
        if len(parts) >= 4:
            return parts[2], parts[3], is_vec  # session_id, hash, is_vec
        return None, None, is_vec

    def export_records(self) -> List[Dict[str, Any]]:
        """
        One record per base key (no :vec/:meta), including:
        session_id, key, key_hash, has_vector, created_at, last_accessed,
        prompt, response, label, source
        """
        records: List[Dict[str, Any]] = []
        try:
            for k in self.client.scan_iter(match="chat:v1:*"):
                # skip meta & vector keys
                if (isinstance(k, bytes) and (k.endswith(b":meta") or k.endswith(b":vec"))) \
                   or (isinstance(k, str) and (k.endswith(":meta") or k.endswith(":vec"))):
                    continue

                key_str = k.decode() if isinstance(k, bytes) else str(k)

                # hide persisted config row in rich export
                if key_str == "chat:v1:config":
                    continue

                session_id, key_hash, _ = self._parse_key(key_str)

                val = self.get(key_str)
                if isinstance(val, dict):
                    prompt = val.get("prompt")
                    response = val.get("response")
                    label = val.get("label")
                    source = val.get("source")
                else:
                    prompt, response, label, source = None, val, None, None

                meta_raw = self.client.get(f"{key_str}:meta")
                meta = self._decompress(meta_raw) if meta_raw else {}

                vec_key = f"{key_str}:vec"
                has_vector = self.client.exists(vec_key) == 1

                rec = {
                    "session_id": session_id,
                    "key": key_str,
                    "key_hash": key_hash,
                    "has_vector": bool(has_vector),
                    "created_at": meta.get("created_at"),
                    "last_accessed": meta.get("last_accessed"),
                    "prompt": prompt,
                    "response": response,
                    "label": label,
                    "source": source,
                }
                records.append(rec)
        except redis.exceptions.RedisError as e:
            print(f"[RedisCache] EXPORT_RECORDS failed: {e}")
        return records

    def export_full_csv(self) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        header = [
            "session_id","key","key_hash","has_vector",
            "created_at","last_accessed","prompt","response","label","source"
        ]
        writer.writerow(header)
        for rec in self.export_records():
            writer.writerow([
                rec.get("session_id"),
                rec.get("key"),
                rec.get("key_hash"),
                rec.get("has_vector"),
                rec.get("created_at"),
                rec.get("last_accessed"),
                rec.get("prompt"),
                rec.get("response"),
                rec.get("label"),
                rec.get("source"),
            ])
        return output.getvalue()


# --------- Semantic Caching Logic ---------
_embedder_instance: Optional[SentenceTransformer] = None

def get_embedder() -> SentenceTransformer:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder_instance

def generate_key(session_id: str, message: str) -> str:
    return f"chat:{session_id}:{hashlib.sha256(message.encode()).hexdigest()}"

def embed_text(text: str) -> Optional[np.ndarray]:
    try:
        return np.array(get_embedder().encode([text])[0])
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return None

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if vec1 is None or vec2 is None:
        return 0.0
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot / norm if norm != 0 else 0.0

def find_semantic_match(cache: RedisCache, session_id: str, query_embedding: np.ndarray) -> Optional[str]:
    pattern = f"chat:v1:{session_id}:*:vec"
    best_key, best_score = None, -1.0
    for key in cache.scan_iter(match=pattern):
        key_str = key.decode() if isinstance(key, bytes) else str(key)
        embedding = cache.get(key_str)
        if not embedding:
            continue
        try:
            stored_vec = np.array(embedding, dtype=np.float32)
            score = cosine_similarity(query_embedding, stored_vec)
            if score > best_score:
                best_score, best_key = score, key_str
        except Exception as e:
            print(f"[Compare Error] {e}")
    return best_key if best_score >= settings.SIMILARITY_THRESHOLD else None

def classify_prompt(prompt: str) -> str:
    p = prompt.lower()
    mapping = {
        "AI.NLP.TransformerModels": ["transformer", "bert", "gpt", "llm", "attention mechanism"],
        "AI.ML.ModelTypes": ["classification", "regression", "supervised", "unsupervised"],
        "AI.NLP.SemanticSearch": ["embedding", "vector search"],
        "Finance.Accounting.RevenueRecognition": ["revenue", "recognition", "accrual", "invoice"],
        "Finance.Risk.CreditScoring": ["credit score", "risk", "loan"],
        "Finance.Tax.Compliance": ["tax", "filing"],
        "Medicine.General.Diagnosis": ["diagnosis", "symptom", "treatment"],
        "Medicine.Oncology.TreatmentPlan": ["oncology", "cancer"],
        "Medicine.Admin.PriorAuth": ["insurance", "authorization", "payer"],
        "Tech.Databases.SQL": ["sql", "join", "query", "select", "index"],
        "Tech.Systems.Caching": ["nosql", "redis", "cache", "memory store"],
        "Tech.Backend.API": ["api", "endpoint", "rest", "postman"],
    }
    for label, terms in mapping.items():
        if any(term in p for term in terms):
            return label
    return "General.Uncategorized"

def process_prompt(cache: RedisCache, req: PromptRequest) -> PromptResponse:
    message = req.message or req.prompt
    embedding = embed_text(message) if settings.USE_SEMANTIC_CACHE else None

    # Semantic path
    if embedding is not None:
        match_key = find_semantic_match(cache, req.session_id, embedding)
        if match_key:
            CACHE_HITS.labels(type="semantic").inc()
            base_key = match_key.replace(":vec", "")
            entry = cache.get(base_key)
            if isinstance(entry, dict):
                return PromptResponse(
                    session_id=req.session_id,
                    response=entry.get("response", ""),
                    from_cache=True,
                    similarity=cosine_similarity(embedding, np.array(cache.get(match_key))),
                    label=entry.get("label"),
                )

    # Exact path
    key = generate_key(req.session_id, message)
    cached = cache.get(key)
    if cached:
        CACHE_HITS.labels(type="exact").inc()
        if isinstance(cached, dict):
            return PromptResponse(
                session_id=req.session_id,
                response=cached.get("response", ""),
                from_cache=True,
                similarity=None,
                label=cached.get("label"),
            )

    # Cache miss - return indication that no cache hit occurred
    CACHE_MISSES.inc()
    return PromptResponse(
        session_id=req.session_id, 
        response="", 
        from_cache=False, 
        similarity=None,
        label=None
    )

def store_response(cache: RedisCache, session_id: str, message: str, response: str) -> None:
    """Store a fresh response in the cache after it's generated by the LLM"""
    key = generate_key(session_id, message)
    label = classify_prompt(message)
    cache.set(key, {"prompt": message, "response": response, "label": label, "source": "fresh"})
    
    # Also store embedding if semantic caching is enabled
    if settings.USE_SEMANTIC_CACHE:
        embedding = embed_text(message)
        if embedding is not None:
            cache.set(f"{key}:vec", embedding.tolist())
