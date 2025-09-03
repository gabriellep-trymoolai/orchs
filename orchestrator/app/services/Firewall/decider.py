from __future__ import annotations
import os, time, datetime, threading
from collections import deque
from typing import Dict, Any, Optional, List

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

DEFAULT_POLICY: Dict[str, Any] = {
    "version": 1,
    "defaults": {
        "llm_enabled": True,
        "pii": {"local_min_score": 0.40, "long_text_chars": 1000, "fuse": "local_or_llm", "llm_role": "annotate_and_gate"},
        "secrets": {"entropy_borderline": 3.4, "fuse": "local_gates_llm_annotates", "llm_role": "annotate_only"},
        "toxicity": {"long_text_chars": 800, "fuse": "local_or_llm", "llm_role": "severity_and_categories"},
        "allow": {"org_requires_llm": False, "fuse": "allow_if_any", "llm_role": "gate"},
    },
    "budget": {"daily_llm_calls_cap": 2000, "per_min_llm_calls_cap": 120},
    "circuit_breakers": {"llm_error_rate_pct": 20, "llm_latency_ms_p95": 2500},
    "overrides": {"tenants": {}, "emergency_override": {"enabled": False, "llm_enabled": False}},
}

def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

class PolicyManager:
    def __init__(self, path: Optional[str] = None, base: Optional[Dict[str, Any]] = None):
        self.path = path
        self._policy = dict(base or DEFAULT_POLICY)
        self._mtime = 0.0
        self._lock = threading.Lock()
        self.load()
    def load(self) -> Dict[str, Any]:
        if self.path and os.path.exists(self.path) and yaml is not None:
            with open(self.path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            self._policy = _deep_merge(DEFAULT_POLICY, data)
            self._mtime = os.path.getmtime(self.path)
        return self._policy
    def maybe_reload(self) -> None:
        if not self.path: return
        try:
            m = os.path.getmtime(self.path)
            if m != self._mtime: self.load()
        except Exception:
            pass
    def get(self) -> Dict[str, Any]:
        with self._lock:
            return self._policy
    def ep(self, endpoint: str) -> Dict[str, Any]:
        d = self.get().get("defaults", {})
        return dict(d.get(endpoint, {}))
    def llm_globally_enabled(self) -> bool:
        p = self.get()
        if p.get("overrides", {}).get("emergency_override", {}).get("enabled"):
            return False
        return bool(p.get("defaults", {}).get("llm_enabled", True))
    def budgets(self) -> Dict[str, int]:
        b = self.get().get("budget", {})
        return {"per_min": int(b.get("per_min_llm_calls_cap", 120)), "daily": int(b.get("daily_llm_calls_cap", 2000))}
    def breakers(self) -> Dict[str, int]:
        c = self.get().get("circuit_breakers", {})
        return {"err_pct": int(c.get("llm_error_rate_pct", 20)), "lat_p95": int(c.get("llm_latency_ms_p95", 2500))}

class BudgetGuard:
    def __init__(self, per_min_cap: int, daily_cap: int):
        self.per_min_cap = per_min_cap
        self.daily_cap = daily_cap
        self._minute = int(time.time() // 60)
        self._min_count = 0
        self._day = datetime.date.today()
        self._day_count = 0
        self._lock = threading.Lock()
    def can_call(self) -> bool:
        with self._lock:
            now_min = int(time.time() // 60)
            if now_min != self._minute:
                self._minute, self._min_count = now_min, 0
            today = datetime.date.today()
            if today != self._day:
                self._day, self._day_count = today, 0
            return (self._min_count < self.per_min_cap) and (self._day_count < self.daily_cap)
    def record(self) -> None:
        with self._lock:
            self._min_count += 1
            self._day_count += 1

class HealthMonitor:
    def __init__(self, err_pct_thresh: int, p95_ms_thresh: int):
        self.errs = deque(maxlen=200)
        self.lats = deque(maxlen=200)
        self.err_pct_thresh = err_pct_thresh
        self.p95_ms_thresh = p95_ms_thresh
        self._lock = threading.Lock()
    def can_call(self) -> bool:
        with self._lock:
            if len(self.lats) < 5: return True
            err_rate = (sum(self.errs) / max(len(self.errs), 1)) * 100.0
            p95 = _percentile(list(self.lats), 95)
            return (err_rate <= self.err_pct_thresh) and (p95 <= self.p95_ms_thresh)
    def record(self, latency_ms: float, error: bool) -> None:
        with self._lock:
            self.lats.append(latency_ms)
            self.errs.append(1 if error else 0)

def _percentile(xs: List[float], p: float) -> float:
    if not xs: return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c: return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)

class Decider:
    def __init__(self, pm: PolicyManager, budget: BudgetGuard, health: HealthMonitor, llm_env_enabled: bool):
        self.pm = pm
        self.budget = budget
        self.health = health
        self.llm_env_enabled = llm_env_enabled
    def decide(self, endpoint: str, local_meta: Dict[str, Any]) -> Dict[str, Any]:
        self.pm.maybe_reload()
        ep = self.pm.ep(endpoint)
        if not self.pm.llm_globally_enabled() or not self.llm_env_enabled:
            return {"use_llm": False, "fuse": ep.get("fuse", "local_only"), "why": "llm disabled"}
        text_len = int(local_meta.get("text_len", 0))
        detected = bool(local_meta.get("detected", False))
        conf = float(local_meta.get("confidence", 0.0))
        entropy_list: List[float] = local_meta.get("entropy_list", []) or []
        org_requires_llm = bool(local_meta.get("org_requires_llm", False))
        escalate = False; why = ""
        if endpoint == "pii":
            if conf < float(ep.get("local_min_score", 0.4)): escalate, why = True, f"low_conf({conf})"
            elif not detected and text_len > int(ep.get("long_text_chars", 1000)): escalate, why = True, f"long_text({text_len})"
        elif endpoint == "secrets":
            border = float(ep.get("entropy_borderline", 3.4))
            borderline_hits = sum(1 for e in entropy_list if e >= border and e < (border + 0.25))
            if borderline_hits > 0: escalate, why = True, f"borderline_entropy_hits({borderline_hits})"
        elif endpoint == "toxicity":
            if detected or text_len > int(ep.get("long_text_chars", 800)): escalate, why = True, ("local_flagged" if detected else f"long_text({text_len})")
        elif endpoint == "allow":
            if (not detected) and (org_requires_llm or bool(ep.get("org_requires_llm", False))): escalate, why = True, "no_local_match+org_requires_llm"
        if not escalate:
            return {"use_llm": False, "fuse": ep.get("fuse", "local_only"), "why": "no_escalation"}
        if not self.health.can_call():
            return {"use_llm": False, "fuse": ep.get("fuse", "local_only"), "why": "health_guard"}
        if not self.budget.can_call():
            return {"use_llm": False, "fuse": ep.get("fuse", "local_only"), "why": "budget_guard"}
        return {"use_llm": True, "fuse": ep.get("fuse", "local_only"), "why": why}
    def record_llm_outcome(self, latency_ms: float, error: bool) -> None:
        self.budget.record()
        self.health.record(latency_ms, error)
