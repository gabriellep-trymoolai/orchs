# orchestrator/app/o_gui.py - INTERNAL/ DEBUG ONLY
from __future__ import annotations
import os, time, uuid, asyncio, json
from typing import Optional, Dict
from fastapi import FastAPI, HTTPException, Request, Query
from pydantic import BaseModel

# import 
from orchestrator.app.o_config import orchestrator_config

# Optional buffer manager 
try:
    from orchestrator.app.utils.buffer_manager import OrchestratorBufferManager, buffer_manager
except Exception:
    OrchestratorBufferManager = None
    buffer_manager = None

app = FastAPI(title="O-GUI")

# --- env ---
ORCHESTRATOR_ID        = os.getenv("ORCHESTRATOR_ID", "orch-001")
ORCHESTRATOR_GUI_HOST  = os.getenv("ORCHESTRATOR_GUI_HOST", "0.0.0.0")
ORCHESTRATOR_GUI_PORT  = int(os.getenv("ORCHESTRATOR_GUI_PORT", "8001"))

# Where O-CCS (ws_client) exposes its tiny HTTP (/push_config)
ORCH_HTTP_HOST         = os.getenv("ORCHESTRATOR_HTTP_HOST", "localhost")
ORCH_HTTP_PORT         = int(os.getenv("ORCHESTRATOR_HTTP_PORT", "8754"))

# If you set this, both O-GUI and ws_client will read/write the same file
# e.g. ORCHESTRATOR_CONFIG_FILE=orchestrator/app/data/orchestrator_config.json
CONFIG_FILE_PATH       = os.getenv("ORCHESTRATOR_CONFIG_FILE")

# --- models ---
class OrchestratorCreate(BaseModel):
    org_id: str
    name: str
    location: str
    metadata: dict = {}
    features: dict = {}

class FeatureUpdate(BaseModel):
    features: dict

class LoginIn(BaseModel):
    user_id: str = "orch_admin"


# ---------- helpers ----------
async def _push_config_async():
    """
    Tell ws_client to POST an o_config snapshot up to the controller.
    Uses env ORCHESTRATOR_HTTP_HOST/PORT (not hard-coded).
    """
    url = f"http://{ORCH_HTTP_HOST}:{ORCH_HTTP_PORT}/push_config"
    try:
        # Prefer httpx if installed
        try:
            import httpx
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.post(url)
                if r.status_code >= 400:
                    raise HTTPException(r.status_code, f"push_config failed: {r.text}")
                return
        except Exception:
            pass
        # Fallback to stdlib
        import urllib.request, urllib.error
        def _sync_push():
            req = urllib.request.Request(url, method="POST", data=b"")
            with urllib.request.urlopen(req, timeout=3) as resp:
                resp.read()
        await asyncio.to_thread(_sync_push)
    except Exception as e:
        print(f"[O-GUI] push_config failed: {e}")


async def _active_user_cleanup_loop():
    """Purge stale active users every 60s (30 min idle threshold)."""
    if not buffer_manager:
        return
    while True:
        try:
            buffer_manager.cleanup_expired(timeout_seconds=1800)
        except Exception as e:
            print(f"[O-GUI] cleanup_expired error: {e}")
        await asyncio.sleep(60)


@app.on_event("startup")
async def _startup_bg_tasks():
    asyncio.create_task(_active_user_cleanup_loop())


# --- simple status + config  ---
@app.get("/healthz")
def healthz():
    cfg = orchestrator_config.get_config()
    version = ((cfg or {}).get("metadata") or {}).get("version", "unknown")
    return {
        "ok": True,
        "service": "o-gui",
        "orchestrator_id": ORCHESTRATOR_ID,
        "version": version,
        "ts": time.time(),
    }

@app.get("/status")
def status():
    cfg = orchestrator_config.get_config()
    return {
        "orchestrator_id": ORCHESTRATOR_ID,
        "ok": True,
        "ts": time.time(),
        "cfg_status": cfg.get("status", "unknown"),
        "last_seen": cfg.get("last_seen"),
    }

@app.get("/config")
def get_cfg():
    return orchestrator_config.get_config()

@app.get("/whoami")
def whoami():
    # debug: confirm which file O-GUI is editing
    return {
        "config_file": orchestrator_config.config_file,
        "orchestrator_id": ORCHESTRATOR_ID,
        "http_push_target": f"http://{ORCH_HTTP_HOST}:{ORCH_HTTP_PORT}/push_config",
    }

# --- feature toggles  ---
@app.post("/features")
async def update_features(update: FeatureUpdate):
    cfg = orchestrator_config.get_config()
    cfg.setdefault("features", {})
    cfg["features"].update(update.features)
    orchestrator_config.update_config(cfg)
    # Non-blocking push to controller via ws_client
    asyncio.create_task(_push_config_async())
    return {"status": "updated", "features": cfg["features"]}

# --- demo login & buffers ---
@app.post("/login")
def login(body: LoginIn):
    if buffer_manager:
        buffer_manager.update_active_user(
            user_id=body.user_id, orch_id=ORCHESTRATOR_ID, metadata={"via": "o_gui"}
        )
    return {"status": "logged_in", "user": body.user_id}

# --- prompt processing demo ONLY (add agent) ---
class PromptIn(BaseModel):
    user_id: str = "demo-user"
    prompt: str

def _simulate_agent_response(prompt: str) -> dict:
    return {
        "text": f"[demo] reply to: {prompt}",
        "tokens_in": len(prompt.split()),
        "tokens_out": 8,
        "latency_ms": 1200,
    }

@app.post("/process_prompt")
async def process_prompt(p: PromptIn, request: Request):
    prompt_id = str(uuid.uuid4())
    meta = {"orchestrator_id": ORCHESTRATOR_ID, "received_ts": time.time()}

    if buffer_manager:
        try:
            buffer_manager.add_prompt(
                prompt_id=prompt_id,
                user_id=p.user_id,
                prompt=p.prompt,
                response=None,
                metadata=meta,
            )
        except Exception as e:
            print(f"[O-GUI] buffer add_prompt error: {e}")

    asyncio.create_task(_run_agent(prompt_id, p.user_id, p.prompt, meta))

    # Optional: keep the HTTP socket open
    try:
        raw_delay = request.query_params.get("delay", "0")
        delay = max(0.0, min(10.0, float(raw_delay)))
        if delay:
            await asyncio.sleep(delay)
    except ValueError:
        raise HTTPException(status_code=400, detail="delay must be a number")

    simulate = request.query_params.get("simulate", "0").lower() in ("1","true","yes")
    if simulate:
        await asyncio.sleep(0.2)
        sim = _simulate_agent_response(p.prompt)
        try:
            if buffer_manager and hasattr(buffer_manager, "update_prompt_response"):
                buffer_manager.update_prompt_response(prompt_id, sim)
        except Exception as e:
            print(f"[O-GUI] buffer update error: {e}")
        return {"prompt_id": prompt_id, "status": "accepted", "response": sim}

    return {"prompt_id": prompt_id, "status": "accepted"}

@app.get("/prompt/{prompt_id}")
def get_prompt(prompt_id: str):
    if not buffer_manager:
        raise HTTPException(500, "buffer manager not available")
    item = buffer_manager.get_prompt(prompt_id)
    if not item:
        raise HTTPException(404, "unknown prompt_id")
    return item

@app.get("/recent_prompts")
def recent_prompts(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    if not buffer_manager:
        raise HTTPException(500, "buffer manager not available")
    return buffer_manager.get_recent_prompts(limit=limit, offset=offset)

@app.get("/buffer_stats")
def buffer_stats():
    if not buffer_manager:
        raise HTTPException(500, "buffer manager not available")
    return buffer_manager.get_stats()

@app.get("/recent_tasks")
def recent_tasks(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    if not buffer_manager:
        raise HTTPException(500, "buffer manager not available")
    return buffer_manager.get_recent_tasks(limit=limit, offset=offset)

@app.get("/active_users")
def active_users():
    if not buffer_manager:
        raise HTTPException(500, "buffer manager not available")
    return buffer_manager.get_active_users()

@app.get("/active_users/{user_id}")
def active_user(user_id: str):
    if not buffer_manager:
        raise HTTPException(500, "buffer manager not available")
    item = buffer_manager.get_active_user(user_id)
    if not item:
        raise HTTPException(404, "unknown user_id")
    return item


# --- run directly  ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=ORCHESTRATOR_GUI_HOST, port=ORCHESTRATOR_GUI_PORT)
