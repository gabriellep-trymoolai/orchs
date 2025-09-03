from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os, json, re, math, string, time
from pathlib import Path

# --- load .env ---
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)
except Exception:
    pass

# ========================= PII (Presidio + spaCy) =========================
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider

def ensure_spacy_model(model_name: str = "en_core_web_sm") -> None:
    try:
        import importlib; importlib.import_module(model_name)
    except Exception:
        from spacy.cli import download; download(model_name)
ensure_spacy_model("en_core_web_sm")

provider = NlpEngineProvider(
    nlp_configuration={"nlp_engine_name":"spacy","models":[{"lang_code":"en","model_name":"en_core_web_sm"}]}
)
nlp_engine = provider.create_engine()
registry = RecognizerRegistry(); registry.load_predefined_recognizers()

# Stronger SSN pattern
ssn_pattern = Pattern(
    name="US_SSN_PATTERN",
    regex=r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b",
    score=0.8,
)
registry.add_recognizer(PatternRecognizer(
    supported_entity="US_SSN",
    patterns=[ssn_pattern],
    context=["ssn","social security","social-security","social sec"]
))
analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry, supported_languages=["en"])
TARGET_ENTITIES = ["EMAIL_ADDRESS","US_SSN","PHONE_NUMBER","CREDIT_CARD","IP_ADDRESS"]
SCORE_THRESHOLD = 0.4

# ========================= Secrets (regex + entropy) ======================

def _redact(s: str) -> str:
    if not s: return ""
    return "*"*len(s) if len(s)<=8 else s[:4] + "*"*(len(s)-8) + s[-4:]


def _entropy(s: str) -> float:
    if not s: return 0.0
    freq={ch:s.count(ch) for ch in set(s)}
    return -sum((c/len(s))*math.log2(c/len(s)) for c in freq.values())


SECRET_PATTERNS = [
    ("AWS Access Key ID",
     re.compile(r"(?<![A-Z0-9])(AKIA|ASIA|AIDA|AGPA|ANPA|AROA|AIPA)[A-Z0-9]{16}(?![A-Z0-9])"), 0),
    ("AWS Secret Access Key",
     re.compile(r"(?i)\baws[_-]?secret[_-]?access[_-]?key\b\s*[:=]\s*([A-Za-z0-9/\+=]{40})"), 1),
    ("GitHub Token",         re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36}\b"), 0),
    ("Slack Token",          re.compile(r"\bxox[abprs]-[0-9A-Za-z-]{10,}\b"), 0),
    ("Google API Key",       re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b"), 0),
    ("OpenAI API Key",       re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), 0),
    ("Stripe Live Key",      re.compile(r"\b(?:sk_live|rk_live)_[A-Za-z0-9]{24,}\b"), 0),
    ("Twilio Account SID",   re.compile(r"\bAC[0-9a-fA-F]{32}\b"), 0),
    ("Twilio Auth Token",    re.compile(r"\b[0-9a-fA-F]{32}\b"), 0),
    ("Private Key Block",    re.compile(r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----"), 0),
]

# ---- LLM secret scrubber (server-side final safeguard) ----
SUSPECT_KEYS = {
    "redacted","value","secret","token","key","api_key","access_key",
    "client_secret","private_key","password","credential"
}

def _scrub_llm_secrets(obj):
    """Recursively mask any suspect fields in the LLM JSON (first 4 + **** + last 4)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, str) and k.lower() in SUSPECT_KEYS:
                out[k] = _redact(v)
            else:
                out[k] = _scrub_llm_secrets(v)
        return out
    if isinstance(obj, list):
        return [_scrub_llm_secrets(x) for x in obj]
    return obj

def scan_secrets_regex(text: str, entropy_threshold: float = 3.5):
    findings=[]
    for name, pattern, grp in SECRET_PATTERNS:
        for m in pattern.finditer(text):
            full = m.group(grp) if (grp and (m.lastindex or 0) >= grp) else m.group(0)
            findings.append({"detector":name,"redacted":_redact(full),
                             "entropy":round(_entropy(full),3),"start":m.start(),"end":m.end()})
    for m in re.finditer(r"\b[A-Za-z0-9/\+=]{20,}\b", text):
        s=m.group(0); ent=_entropy(s)
        already=any(d["start"]<=m.start()<=d["end"] for d in findings)
        if ent>=entropy_threshold and not already:
            findings.append({"detector":"High-Entropy String","redacted":_redact(s),
                             "entropy":round(ent,3),"start":m.start(),"end":m.end()})
    findings.sort(key=lambda x:(x["start"],-(x["end"]-x["start"])) )
    dedup=[]
    for f in findings:
        if not any(f["start"]>=d["start"] and f["end"]<=d["end"] for d in dedup):
            dedup.append(f)
    return dedup

# ========================= Toxicity (better_profanity only) ===============
CUSTOM_TOXIC_WORDS = {
    "hate","hateful","disgusting","idiot","stupid","dumb","moron","loser",
    "trash","garbage","worthless","ugly"
}
_BP_READY = False


def _init_better_profanity():
    global _BP_READY
    if _BP_READY:
        return
    from better_profanity import profanity
    profanity.load_censor_words()
    try:
        profanity.add_censor_words(list(CUSTOM_TOXIC_WORDS))
    except Exception:
        pass
    _BP_READY = True


_table = str.maketrans({c:" " for c in string.punctuation})


def _simple_tokens(s: str):
    return (s or "").translate(_table).lower().split()


# ========================= LLM backup (OpenAI) ============================
LLM_BACKUP_ENABLED = os.getenv("LLM_BACKUP_ENABLED","0") == "1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL","gpt-4o-mini")
OPENAI_API_URL = os.getenv("OPENAI_API_URL","https://api.openai.com/v1/chat/completions")



def _openai_json(system_prompt: str, user_payload: dict) -> dict:
    import requests
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role":"system","content": system_prompt},
            {"role":"user","content": json.dumps(user_payload, ensure_ascii=False)}
        ],
        "temperature": 0,
        "response_format": {"type":"json_object"},
    }
    t0 = time.time()
    r = requests.post(OPENAI_API_URL, headers=headers, json=body, timeout=45)
    latency_ms = (time.time() - t0) * 1000.0
    try:
        data = r.json()
    except Exception:
        raise HTTPException(status_code=502, detail=f"OpenAI HTTP {r.status_code}")
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=str(data))
    content = data["choices"][0]["message"]["content"]
    return json.loads(content), latency_ms


PII_SYS = (
    "You are a privacy classifier. Reply ONLY in JSON.\n"
    'Schema: {"contains_pii": boolean, "findings":[{"entity_type": string, "text": string}]}\n'
    "Consider emails, phone numbers, SSNs, credit cards, IPs. No extra text."
)
SECRETS_SYS = (
    "You are a secret detector. Reply ONLY in JSON.\n"
    'Schema: {"contains_secrets": boolean, "items":[{"type": string, "redacted": string}]}\n'
    "Consider API keys, tokens, access keys, private keys.\n"
    "IMPORTANT: NEVER output a full secret. Always mask as first 4 chars + **** + last 4.\n"
    "No extra text."
)
ALLOW_SYS = (
    "You are an allowlist matcher. Reply ONLY in JSON.\n"
    'Schema: {"allowed": boolean, "matched_topic": string|null}\n'
    "Return allowed=true if any topic is contained (case-insensitive). No extra text."
)
TOX_SYS = (
    "You are a content-safety classifier. Reply ONLY in JSON.\n"
    'Schema: {"toxic": boolean, "severity":"low|medium|high", "categories": ["harassment"|"threat"|"identity_attack"|"sexual"|"self_harm"|"other"...], "reason": string}\n'
    "Be strict for harassment and threats. No extra text."
)


# ========================= FastAPI & models ===============================
app = FastAPI(title="Mini Firewall (Unified 4 Endpoints, Policy-Driven)")


class TextRequest(BaseModel):
    text: str


class AllowlistRequest(BaseModel):
    text: str
    topics: Optional[List[str]] = None


@app.get("/")
def root():
    return {"ok": True, "endpoints": ["/scan/pii","/scan/secrets","/scan/toxicity","/scan/allow"]}


# ---------- Local runners with meta ----------

def _pii_local(text: str):
    try:
        results = analyzer.analyze(text=text, language="en", entities=TARGET_ENTITIES, score_threshold=SCORE_THRESHOLD)
        filtered = [r for r in results if r.entity_type in TARGET_ENTITIES and r.score >= SCORE_THRESHOLD]
        findings = [{
            "entity_type": r.entity_type,
            "score": float(round(r.score, 3)),
            "start": r.start,
            "end": r.end,
            "text": text[r.start:r.end]
        } for r in filtered]
        findings = sorted(findings, key=lambda x: (x["start"], -(x["end"] - x["start"])) )
        dedup = []
        for f in findings:
            if not any(f["start"] >= d["start"] and f["end"] <= d["end"] for d in dedup):
                dedup.append(f)
        max_score = max([f.get("score", 0.0) for f in dedup], default=0.0)
        return {"contains_pii": bool(dedup), "findings": dedup, "confidence": max_score}
    except Exception as e:
        return {"contains_pii": False, "findings": [], "confidence": 0.0, "error": str(e)}


def _secrets_local(text: str):
    f = scan_secrets_regex(text)
    norm = [{"type": x.get("detector"), "redacted": x.get("redacted"), "entropy": x.get("entropy"), "start": x.get("start"), "end": x.get("end")} for x in f]
    max_entropy = max([x.get("entropy", 0.0) for x in f], default=0.0)
    entropy_list = [float(x.get("entropy", 0.0)) for x in f]
    return {"contains_secrets": bool(f), "findings": norm, "max_entropy": max_entropy, "entropy_list": entropy_list}


def _toxicity_local(text: str):
    try:
        from better_profanity import profanity
        _init_better_profanity()
        flagged = bool(profanity.contains_profanity(text or ""))
        return {"contains_toxicity": flagged}
    except Exception as e:
        return {"contains_toxicity": False, "error": str(e)}


def _allow_local(text: str, topics: Optional[List[str]]):
    topics = topics or []
    lowered = (text or "").lower()
    matched = None
    for t in topics:
        if (t or "").lower() in lowered:
            matched = t
            break
    return {"allowed": bool(matched), "matched_topic": matched}


# ---------- Policy + Decider ----------
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decider import PolicyManager, BudgetGuard, HealthMonitor, Decider, DEFAULT_POLICY

POLICY_PATH = os.getenv("POLICY_PATH", str(Path(__file__).with_name("policy.yaml")))
pm = PolicyManager(POLICY_PATH, base=DEFAULT_POLICY)
_b = pm.budgets(); _c = pm.breakers()
_budget = BudgetGuard(per_min_cap=_b["per_min"], daily_cap=_b["daily"])
_health = HealthMonitor(err_pct_thresh=_c["err_pct"], p95_ms_thresh=_c["lat_p95"])
_decider = Decider(pm, _budget, _health, llm_env_enabled=LLM_BACKUP_ENABLED and bool(OPENAI_API_KEY))


# ---------- Fusion helpers ----------

def _fuse(endpoint: str, fuse: str, local: dict, llm: Optional[dict]):
    if endpoint == "pii":
        local_det = bool(local.get("contains_pii"))
        llm_det = bool(llm and llm.get("contains_pii"))
        if fuse == "local_only":
            return local_det
        return local_det or llm_det
    if endpoint == "secrets":
        # local decides; LLM annotates only
        return bool(local.get("contains_secrets"))
    if endpoint == "toxicity":
        local_det = bool(local.get("contains_toxicity"))
        llm_det = bool(llm and llm.get("toxic"))
        if fuse == "local_only":
            return local_det
        return local_det or llm_det
    if endpoint == "allow":
        local_ok = bool(local.get("allowed"))
        llm_ok = bool(llm and llm.get("allowed"))
        if fuse == "local_only":
            return local_ok
        # allow_if_any
        return local_ok or llm_ok
    return False


# ========================= Endpoints =============================

@app.post("/scan/pii")
async def scan_pii(req: Request, payload: TextRequest):
    text = payload.text or ""
    local = _pii_local(text)
    meta = {"detected": local.get("contains_pii", False), "confidence": float(local.get("confidence", 0.0)), "text_len": len(text)}
    decision = _decider.decide("pii", meta)

    llm = None; llm_latency = 0.0; llm_err = False
    if decision.get("use_llm"):
        try:
            llm, llm_latency = _openai_json(PII_SYS, {"text": text})
        except Exception:
            llm_err = True
        finally:
            _decider.record_llm_outcome(llm_latency, llm_err)

    final_flag = _fuse("pii", decision.get("fuse", "local_or_llm"), local, llm)
    return {"endpoint": "pii", "strategy": decision, "local": local, "llm": llm, "contains_pii": bool(final_flag)}


@app.post("/scan/secrets")
async def scan_secrets(req: Request, payload: TextRequest):
    text = payload.text or ""
    local = _secrets_local(text)
    meta = {
        "detected": local.get("contains_secrets", False),
        "confidence": float(local.get("max_entropy", 0.0)),
        "entropy_list": list(local.get("entropy_list", [])),
        "text_len": len(text),
    }
    decision = _decider.decide("secrets", meta)

    llm = None
    llm_latency = 0.0
    llm_err = False
    if decision.get("use_llm"):
        try:
            llm, llm_latency = _openai_json(SECRETS_SYS, {"text": text})
            llm = _scrub_llm_secrets(llm)  # enforce masking on model output
        except Exception:
            llm_err = True
        finally:
            _decider.record_llm_outcome(llm_latency, llm_err)

    final_flag = _fuse("secrets", decision.get("fuse", "local_gates_llm_annotates"), local, llm)
    return {
        "endpoint": "secrets",
        "strategy": decision,
        "local": local,
        "llm": llm,
        "contains_secrets": bool(final_flag),
    }

@app.post("/scan/toxicity")
async def scan_toxicity(req: Request, payload: TextRequest):
    text = payload.text or ""
    local = _toxicity_local(text)
    meta = {"detected": local.get("contains_toxicity", False), "confidence": 1.0 if local.get("contains_toxicity") else 0.0, "text_len": len(text)}
    decision = _decider.decide("toxicity", meta)

    llm = None; llm_latency = 0.0; llm_err = False
    if decision.get("use_llm"):
        try:
            llm, llm_latency = _openai_json(TOX_SYS, {"text": text})
        except Exception:
            llm_err = True
        finally:
            _decider.record_llm_outcome(llm_latency, llm_err)

    final_flag = _fuse("toxicity", decision.get("fuse", "local_or_llm"), local, llm)
    return {"endpoint": "toxicity", "strategy": decision, "local": local, "llm": llm, "contains_toxicity": bool(final_flag)}


@app.post("/scan/allow")
async def scan_allow(req: Request, payload: AllowlistRequest):
    text = payload.text or ""
    local = _allow_local(text, payload.topics)
    # org requirement could also come from tenant policy; we keep it simple
    meta = {"detected": local.get("allowed", False), "confidence": 1.0 if local.get("allowed") else 0.0, "text_len": len(text), "org_requires_llm": False}
    decision = _decider.decide("allow", meta)

    llm = None; llm_latency = 0.0; llm_err = False
    if decision.get("use_llm"):
        try:
            llm, llm_latency = _openai_json(ALLOW_SYS, {"text": text, "topics": payload.topics or []})
        except Exception:
            llm_err = True
        finally:
            _decider.record_llm_outcome(llm_latency, llm_err)

    final_flag = _fuse("allow", decision.get("fuse", "allow_if_any"), local, llm)
    matched_topic = local.get("matched_topic") or (llm and llm.get("matched_topic"))
    return {"endpoint": "allow", "strategy": decision, "local": local, "llm": llm, "allowed": bool(final_flag), "matched_topic": matched_topic}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
