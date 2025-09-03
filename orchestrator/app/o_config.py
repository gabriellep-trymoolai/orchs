# orchestrator/app/o_config.py
from __future__ import annotations
import json, os, tempfile, threading, time
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pathlib import Path
import hashlib

try:
    import fcntl  # type: ignore
    _HAVE_FCNTL = True
except Exception:
    fcntl = None
    _HAVE_FCNTL = False

_APP_DIR = Path(__file__).resolve().parent
_DEFAULT_PATH = (
    os.getenv("ORCHESTRATOR_CONFIG_FILE")
    or os.getenv("ORCHESTRATOR_CONFIG_PATH")
    or str(_APP_DIR / "db" / "orchestrator_config.json")
)
Path(os.path.dirname(_DEFAULT_PATH) or ".").mkdir(parents=True, exist_ok=True)
_DEFAULT_ORCH_ID = os.getenv("ORCHESTRATOR_ID", "orch-001")

# Debounce window for disk writes (seconds). Tune via env
_SAVE_MIN_INTERVAL = float(os.getenv("ORCH_CONFIG_SAVE_MIN_INTERVAL_SECONDS", "30"))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(obj: Any) -> bytes:
    # stable representation for hashing/comparison
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _hash_cfg(obj: Any) -> str:
    return hashlib.md5(_canonical_bytes(obj)).hexdigest()


class OrchestratorConfigManager:
    def __init__(self, config_file: str = _DEFAULT_PATH):
        self.config_file = config_file
        self._rw = threading.RLock()
        self._file_lock = threading.Lock()
        self._cfg: Dict[str, Any] = {}
        self._cfg_hash: Optional[str] = None
        self._last_save_ts: float = 0.0
        self._load()
        self._ensure_minimums()

    # ------------ file locking helpers ------------
    def _acquire_file_lock(self, fh, timeout_sec: int = 5) -> bool:
        if not _HAVE_FCNTL:
            return True
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore
                return True
            except Exception:
                time.sleep(0.05)
        return False

    def _release_file_lock(self, fh) -> None:
        if not _HAVE_FCNTL:
            return
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)  # type: ignore
        except Exception:
            pass

    # ------------ load / save ------------
    def _load(self) -> Dict[str, Any]:
        with self._file_lock:
            if not os.path.exists(self.config_file):
                cfg = self._default_config()
                self._save_atomic(cfg, first_time=True)
                self._cfg = cfg
                self._cfg_hash = _hash_cfg(cfg)
                print(f"[O-CONFIG] Created default at {self.config_file}")
                return cfg
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    if self._acquire_file_lock(f):
                        try:
                            data = f.read().strip()
                            self._cfg = json.loads(data) if data else self._default_config()
                        finally:
                            self._release_file_lock(f)
                    else:
                        print(f"[O-CONFIG] lock timeout; using in-memory")
                self._cfg_hash = _hash_cfg(self._cfg)
                return self._cfg
            except Exception as e:
                print(f"[O-CONFIG] load error: {e}; using defaults")
                self._cfg = self._default_config()
                self._cfg_hash = _hash_cfg(self._cfg)
                return self._cfg

    def _save_atomic(self, cfg: Dict[str, Any], first_time: bool = False) -> None:
        # always called with file_lock held by caller
        cfg.setdefault("metadata", {})
        cfg["metadata"]["last_updated"] = _utc_now_iso()
        os.makedirs(os.path.dirname(self.config_file) or ".", exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=os.path.dirname(self.config_file) or ".",
            prefix=os.path.basename(self.config_file) + ".",
            suffix=".tmp",
            delete=False,
        ) as tf:
            tmp = tf.name
            if self._acquire_file_lock(tf):
                try:
                    json.dump(cfg, tf, indent=2, sort_keys=True, ensure_ascii=False)
                    tf.flush()
                    os.fsync(tf.fileno())
                finally:
                    self._release_file_lock(tf)
            else:
                json.dump(cfg, tf, indent=2, sort_keys=True, ensure_ascii=False)
                tf.flush()
                os.fsync(tf.fileno())
        os.replace(tmp, self.config_file)
        print(f"[O-CONFIG] saved {self.config_file}" if not first_time else f"[O-CONFIG] initialized {self.config_file}")

    def _save_if_changed(self, new_cfg: Dict[str, Any], reason: str = "") -> bool:
        """
        Change-aware + throttled save.
        Returns True if a disk write occurred.
        """
        with self._file_lock:
            new_hash = _hash_cfg(new_cfg)
            if new_hash == self._cfg_hash:
                # no change → skip disk write
                return False

            now = time.time()
            self._cfg = dict(new_cfg)   # update memory so readers see latest
            self._cfg_hash = new_hash

            if now - self._last_save_ts < _SAVE_MIN_INTERVAL:
                # debounce: skip writing to disk for now
                # (we still updated memory and last_updated will be written next flush)
                return False

            self._save_atomic(self._cfg)
            self._last_save_ts = now
            return True

    # ------------ defaults / normalization ------------
    def _default_config(self) -> Dict[str, Any]:
        return {
            "service_type": "orchestrator",
            "orchestrator_id": _DEFAULT_ORCH_ID,
            "organization": {"name": "Default Orchestrator", "location": "localhost"},
            "features": {},
            "metadata": {
                "version": "1.0.0",
                "created": _utc_now_iso(),
                "last_updated": _utc_now_iso(),
                "capabilities": [],
                "ssl_enabled": False,
            },
            "status": "inactive",
            "last_seen": None,
        }

    def _ensure_minimums(self) -> None:
        with self._rw:
            c = self._cfg
            c.setdefault("orchestrator_id", _DEFAULT_ORCH_ID)
            c.setdefault("organization", {})
            c["organization"].setdefault("name", "Unknown Orchestrator")
            c["organization"].setdefault("location", "Unknown")
            c.setdefault("features", {})
            c.setdefault("metadata", {})
            c.setdefault("status", "inactive")
            c.setdefault("last_seen", None)

    # ------------ public API ------------
    def get_config(self) -> Dict[str, Any]:
        with self._rw:
            return dict(self._cfg)

    def update_config(self, new_cfg: Dict[str, Any]) -> None:
        with self._rw:
            merged = dict(new_cfg or {})
            self._ensure_minimums()
            wrote = self._save_if_changed(merged, reason="update_config")
            if not wrote:
                # quiet: only log when it actually wrote
                pass

    # provisioning/snapshots to avoid unconditional saves
    def update_config_if_changed(self, new_cfg: Dict[str, Any], reason: str = "") -> bool:
        with self._rw:
            merged = dict(new_cfg or {})
            self._ensure_minimums()
            return self._save_if_changed(merged, reason=reason or "update_if_changed")

    def patch_config(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        with self._rw:
            base = dict(self._cfg)
            for k, v in (patch or {}).items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    base[k].update(v)
                else:
                    base[k] = v
            self._ensure_minimums()
            self._save_if_changed(base, reason="patch_config")
            self._cfg = base
            return dict(self._cfg)

    def reload_config(self) -> Dict[str, Any]:
        with self._rw:
            self._load()
            self._ensure_minimums()
            print(f"[O-CONFIG] reloaded from {self.config_file}")
            return dict(self._cfg)

    def touch_presence(self, status: Optional[str] = None) -> Dict[str, Any]:
        with self._rw:
            if status is not None:
                self._cfg["status"] = status
            self._cfg["last_seen"] = _utc_now_iso()
            # presence writes can piggyback on debounce (dispatch already throttles)
            self._save_if_changed(self._cfg, reason="presence")
            return dict(self._cfg)


# must be an object, not a tuple
orchestrator_config = OrchestratorConfigManager(_DEFAULT_PATH)

__all__ = ["OrchestratorConfigManager", "orchestrator_config"]
