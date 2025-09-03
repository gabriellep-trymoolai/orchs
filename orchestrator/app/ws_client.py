# services/orchestrator/app/ws_client.py
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import signal
import socket
import ssl
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Optional

print("[O-CCS] booting ws_client…")


def run() -> None:
    """Main entrypoint for the orchestrator WS client."""
    # --------------------------- logging ---------------------------
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="[O-CCS] %(asctime)s %(levelname)s: %(message)s",
    )
    logger = logging.getLogger("o_ccs")

    # --------------------------- env / dotenv ---------------------------
    app_dir = Path(__file__).resolve().parent
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(app_dir.parents[0] / ".env")
        load_dotenv(app_dir / ".env", override=False)
    except Exception:
        pass

    controller_ws_url = os.getenv("CONTROLLER_WS_URL", "").strip()
    controller_host = os.getenv("CONTROLLER_HOST", "localhost").strip()
    controller_port = int(os.getenv("CONTROLLER_PORT", "8765"))
    ws_path = os.getenv("CONTROLLER_WS_PATH", "").strip()
    ws_path = ("/" + ws_path.lstrip("/")) if ws_path else ""

    orchestrator_id = os.getenv("ORCHESTRATOR_ID", "orch-001")

    keepalive_interval = int(os.getenv("WEBSOCKET_KEEPALIVE_INTERVAL", "10"))
    orch_http_enabled = os.getenv("ORCHESTRATOR_HTTP_ENABLED", "true").lower() in ("1", "true", "yes")
    orch_http_port = int(os.getenv("ORCHESTRATOR_HTTP_PORT", "8754"))
    ssl_disabled = os.getenv("SSL_DISABLED", "true").lower() in ("1", "true", "yes")

    cert_path = Path(os.getenv("CERT_PATH", str(app_dir / "certificates")))
    client_cert = cert_path / "client_cert.pem"
    client_key = cert_path / "client_key.pem"
    ca_cert = cert_path / "ca_cert.pem"

    ocfile = os.getenv("ORCHESTRATOR_CONFIG_FILE") or str(app_dir / "data" / "orchestrator_config.json")

    # Diagnostics
    ws_url_display = controller_ws_url or f"ws://{controller_host}:{controller_port}{ws_path}"
    print(f"[O-CCS] WS URL => {ws_url_display}")
    print(f"[O-CCS] ORCHESTRATOR_ID => {orchestrator_id}")
    print(f"[O-CCS] ORCH_HTTP_ENABLED => {orch_http_enabled}  ORCH_HTTP_PORT => {orch_http_port}")
    print(f"[O-CCS] SSL_DISABLED => {ssl_disabled}")
    print(f"[O-CCS] CONFIG_FILE => {ocfile}")
    print(
        "[O-CCS] CERT_PATH => %s  (client_cert? %s, client_key? %s, ca? %s)"
        % (cert_path, client_cert.exists(), client_key.exists(), ca_cert.exists())
    )

    # --------------------------- module imports ---------------------------
    try:
        orchestrator_config = import_module("orchestrator.app.o_config").orchestrator_config
    except Exception as e:
        logger.error(f"Failed to import o_config: {e}")
        raise

    try:
        dispatch_incoming = import_module("orchestrator.app.utils.dispatch").dispatch_incoming
    except Exception as e:
        logger.warning(f"dispatch import failed (lazy retry later): {e}")
        dispatch_incoming = None  # type: ignore

    # --------------------------- helpers ---------------------------
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def local_ip() -> str:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "0.0.0.0"

    def build_ssl_context() -> Optional[ssl.SSLContext]:
        if ssl_disabled:
            return None
        ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        if ca_cert.exists():
            try:
                ctx.load_verify_locations(str(ca_cert))
                ctx.verify_mode = ssl.CERT_REQUIRED
                ctx.check_hostname = True
                logger.info("TLS: CA verification enabled")
            except Exception as e:
                logger.warning(f"TLS: failed to load CA, falling back to no-verify: {e}")
                ctx.verify_mode = ssl.CERT_NONE
                ctx.check_hostname = False
        else:
            ctx.verify_mode = ssl.CERT_NONE
            ctx.check_hostname = False
            logger.warning("TLS: CA missing; disabling verification (dev)")

        if client_cert.exists() and client_key.exists():
            try:
                ctx.load_cert_chain(str(client_cert), str(client_key))
                logger.info("TLS: loaded client mTLS certs")
            except Exception as e:
                logger.error(f"TLS: client cert load failed: {e}")
        return ctx

    def build_ws_uri(ssl_ctx: Optional[ssl.SSLContext]) -> str:
        if controller_ws_url:
            return controller_ws_url
        scheme = "wss" if ssl_ctx else "ws"
        return f"{scheme}://{controller_host}:{controller_port}{ws_path}"

    def build_handshake(cfg: Dict[str, Any], ssl_ctx: Optional[ssl.SSLContext]) -> Dict[str, Any]:
        org = (cfg or {}).get("organization") or {}
        features = (cfg or {}).get("features") or {}
        orch_id = (cfg or {}).get("orchestrator_id") or orchestrator_id
        name = org.get("name") or f"Organization {orch_id}"
        location = org.get("location") or "unknown"
        version = (cfg.get("metadata", {}) or {}).get("version", "1.0.0")
        return {
            "type": "handshake",
            "service": "orchestrator",
            "data": {
                "orchestrator_id": orch_id,
                "metadata": {
                    "version": version,
                    "hostname": socket.gethostname(),
                    "ip": local_ip(),
                    "ssl_enabled": bool(ssl_ctx),
                    "name": name,
                    "location": location,
                    "features": features,
                },
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

    # --------------------------- state ---------------------------
    connected: bool = False
    counters: Dict[str, int] = {"keepalives": 0, "rx": 0, "tx": 0}
    _ws = None  # type: ignore[assignment]

    # --------------------------- tasks ---------------------------
    async def send_keepalive(ws, stop_evt: asyncio.Event) -> None:
        try:
            while not stop_evt.is_set():
                frame = {"type": "i_am_alive", "timestamp": now_iso()}
                await ws.send(json.dumps(frame))
                counters["keepalives"] += 1
                counters["tx"] += 1
                logging.info("[O-CCS] i_am_alive sent #%s", counters["keepalives"])
                await asyncio.sleep(keepalive_interval)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logging.error("keepalive error: %s", e)

    async def receive_loop(ws) -> None:
        nonlocal dispatch_incoming
        if dispatch_incoming is None:
            try:
                dispatch_incoming = import_module("orchestrator.app.utils.dispatch").dispatch_incoming
            except Exception as e:
                logging.error("dispatch import still failing, closing socket: %s", e)
                try:
                    await ws.close()
                finally:
                    return
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logging.warning("received non-JSON frame; skipping")
                    continue
                counters["rx"] += 1
                try:
                    dispatch_incoming(msg, orchestrator_id, orchestrator_config)  # type: ignore
                except Exception as e:
                    logging.error("dispatch_incoming error: %s", e)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logging.error("receive_loop error: %s", e)

    async def connect() -> None:
        nonlocal connected, _ws
        import websockets  # lazy

        ssl_ctx = build_ssl_context()
        uri = build_ws_uri(ssl_ctx)
        attempt = 0

        while True:
            attempt += 1
            backoff = min(1 * (2 ** min(attempt, 3)), 15) + random.uniform(0, 0.8)
            try:
                logging.info("Connecting to %s (attempt %s)", uri, attempt)
                try:
                    websocket = await asyncio.wait_for(
                        websockets.connect(
                            uri,
                            ssl=ssl_ctx,
                            ping_interval=keepalive_interval * 1.5,  # control pings (client → server)
                            ping_timeout=keepalive_interval,
                            max_queue=64,
                        ),
                        timeout=10,
                    )
                except asyncio.TimeoutError:
                    logging.warning("connection timeout")
                    await asyncio.sleep(backoff)
                    continue
                except Exception as e:
                    logging.error("connect failed: %s", e)
                    await asyncio.sleep(backoff)
                    continue

                # Build + send handshake
                try:
                    cfg = orchestrator_config.get_config()
                except Exception as e:
                    logging.warning("config read failed; continuing with empty: %s", e)
                    cfg = {}

                try:
                    await websocket.send(json.dumps(build_handshake(cfg, ssl_ctx)))
                    counters["tx"] += 1
                except Exception as e:
                    logging.error("failed sending handshake: %s", e)
                    await websocket.close()
                    await asyncio.sleep(backoff)
                    continue

                # Wait for explicit handshake_ack 
                try:
                    raw = await asyncio.wait_for(websocket.recv(), timeout=10)
                    ack = json.loads(raw)
                    if ack.get("type") != "handshake_ack":
                        raise RuntimeError(f"expected handshake_ack, got {ack.get('type')}")
                    logging.info("[O-CCS] handshake_ack received")
                except Exception as e:
                    logging.error("handshake ack error: %s", e)
                    await websocket.close()
                    await asyncio.sleep(backoff)
                    continue

                connected = True
                _ws = websocket

                stop_evt = asyncio.Event()
                ka_task = asyncio.create_task(send_keepalive(websocket, stop_evt))
                rx_task = asyncio.create_task(receive_loop(websocket))

                done, pending = await asyncio.wait(
                    {ka_task, rx_task}, return_when=asyncio.FIRST_COMPLETED
                )
                for t in pending:
                    t.cancel()
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass

                connected = False
                _ws = None
                logging.info("connection ended; retrying shortly")
                await asyncio.sleep(3)

            except asyncio.CancelledError:
                logging.info("connect loop cancelled")
                break
            except Exception as e:
                logging.error("unexpected connect-loop error: %s", e)
                await asyncio.sleep(backoff)

    # --------------------------- internal HTTP  ---------------------------
    async def start_http_server() -> None:
        try:
            from fastapi import FastAPI, HTTPException  # type: ignore
            import uvicorn  # type: ignore
        except Exception as e:
            logging.error("HTTP deps missing: %s", e)
            return

        app = FastAPI(title="O-CCS Internal")

        @app.get("/healthz")
        def healthz():
            return {
                "ok": True,
                "orchestrator_id": orchestrator_id,
                "connected": connected,
                "keepalives_sent": counters["keepalives"],
                "frames_rx": counters["rx"],
                "frames_tx": counters["tx"],
                "ts": now_iso(),
            }

        @app.get("/config")
        def get_config():
            try:
                return orchestrator_config.get_config()
            except Exception as e:
                raise HTTPException(500, f"config read error: {e}")

        config = uvicorn.Config(app, host="0.0.0.0", port=orch_http_port, log_level="info", loop="asyncio")
        server = uvicorn.Server(config)
        await server.serve()

    # --------------------------- run loop ---------------------------
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Graceful stop
    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, loop.stop)
        except Exception:
            pass

    if orch_http_enabled:
        loop.create_task(start_http_server())
        logging.info("Internal HTTP at http://0.0.0.0:%s", orch_http_port)

    try:
        loop.run_until_complete(connect())
    finally:
        tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        loop.close()


if __name__ == "__main__":
    run()
