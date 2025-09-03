# services/orchestrator/app/utils/dispatch.py
from __future__ import annotations
import os, time, uuid, logging
from typing import Any, Dict, Optional
from datetime import datetime, timezone

try:
    from orchestrator.app.utils.buffer_manager import buffer_manager
except Exception:
    buffer_manager = None

from .provisioning import apply_provisioning

logger = logging.getLogger("o_dispatch")

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# --- throttle presence writes ---
_TOUCH_MIN_SECS = float(os.getenv("O_CONFIG_TOUCH_MIN_SECS", "30"))
_last_touch_write_at: float = 0.0

def _persist_last_seen(cfg_mgr) -> None:
    """Persist lightweight presence at most every _TOUCH_MIN_SECS."""
    global _last_touch_write_at, _TOUCH_MIN_SECS
    now = time.time()
    if now - _last_touch_write_at < _TOUCH_MIN_SECS:
        return
    try:
        # centralize into cfg_mgr.touch_presence
        cfg_mgr.touch_presence(status="active")
        _last_touch_write_at = now
    except Exception as e:
        logger.warning(f"[O-DISPATCH] persist last_seen failed: {e}")

def _bm_update_active_user(orch_id: str, meta: Optional[Dict[str, Any]] = None):
    try:
        if buffer_manager:
            buffer_manager.update_active_user(
                user_id=orch_id, orch_id=orch_id, metadata=meta or {}
            )
    except Exception as e:
        logger.warning(f"[O-DISPATCH] active_user update failed: {e}")

def _bm_add_prompt(orch_id: str, prompt_id: str, prompt_text: str, meta: Optional[Dict[str, Any]] = None):
    try:
        if buffer_manager:
            buffer_manager.add_prompt(
                prompt_id=prompt_id,
                user_id=orch_id,
                prompt=prompt_text,
                response=None,
                metadata=meta or {},
            )
    except Exception as e:
        logger.warning(f"[O-DISPATCH] add_prompt failed: {e}")

def _bm_update_prompt(prompt_id: Optional[str], resp_text: str, meta: Optional[Dict[str, Any]] = None):
    if not prompt_id:
        logger.warning("[O-DISPATCH] update_prompt called with no prompt_id")
        return
    try:
        if buffer_manager:
            buffer_manager.update_prompt(
                prompt_id=prompt_id, response=resp_text, metadata=meta or {}
            )
    except Exception as e:
        logger.warning(f"[O-DISPATCH] update_prompt failed: {e}")

def dispatch_incoming(msg: Dict[str, Any], orchestrator_id: str, cfg_mgr) -> None:
    mtype = msg.get("type")

    if mtype == "handshake_ack":
        _bm_update_active_user(orchestrator_id, meta=msg.get("metadata"))
        _persist_last_seen(cfg_mgr)
        logger.info(f"[O-DISPATCH:{orchestrator_id}] handshake_ack")

    elif mtype == "keepalive_ack":
        _bm_update_active_user(orchestrator_id, meta=msg.get("metadata"))
        _persist_last_seen(cfg_mgr)  # throttled
        logger.debug(f"[O-DISPATCH:{orchestrator_id}] keepalive_ack")

    elif mtype == "prompt":
        pid = msg.get("prompt_id") or str(uuid.uuid4())
        _bm_add_prompt(orchestrator_id, pid, msg.get("prompt", ""), {"source": "controller"})
        logger.info(f"[O-DISPATCH:{orchestrator_id}] prompt stored {pid}")

    elif mtype == "prompt_response":
        _bm_update_prompt(msg.get("prompt_id"), msg.get("response", ""), {"source": "controller"})
        logger.info(f"[O-DISPATCH:{orchestrator_id}] prompt_response applied")

    elif mtype == "provisioning":
        changed = apply_provisioning(msg.get("data", {}), cfg_mgr)
        level = logging.INFO if changed else logging.DEBUG
        logger.log(level, f"[O-DISPATCH:{orchestrator_id}] provisioning %s", "applied" if changed else "no-op")


    elif mtype == "command":
        logger.info(f"[O-DISPATCH] command: {msg}")

    elif mtype == "o_config_snapshot_ack":
        logger.debug(f"[O-DISPATCH:{orchestrator_id}] o_config_snapshot_ack")

    else:
        logger.debug(f"[O-DISPATCH] unhandled: {mtype}")
