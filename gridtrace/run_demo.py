#!/usr/bin/env python3
"""
One-command demo launcher: broker + inference + dashboard + scripted publisher.

    python run_demo.py                # scripted tour of all four modes
    python run_demo.py --mode 3       # hold one mode
    python run_demo.py --no-dashboard

Starts everything in one process using threads, so there is a single window to
watch and Ctrl-C stops the lot. If port 1883 is already busy (you run Mosquitto)
the built-in broker steps aside and everything connects to yours instead.
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
import webbrowser

from gridtrace import broker, config


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the whole GridTrace AI prototype locally.")
    ap.add_argument("--mode", type=int, choices=[0, 1, 2, 3], default=None,
                    help="hold one mode instead of running the scripted tour")
    ap.add_argument("--interval", type=float, default=0.7, help="seconds per telemetry frame")
    ap.add_argument("--no-dashboard", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="wipe previous results first")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")

    if not config.BUNDLE_PATH.exists():
        print(f"No trained models at {config.BUNDLE_PATH}\n"
              f"Run this first:   python -m gridtrace.train")
        return 1

    if a.fresh:
        for p in (config.DB_PATH, config.DATA_DIR / "localization.json"):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        for suffix in ("-wal", "-shm"):
            try:
                (config.DB_PATH.parent / (config.DB_PATH.name + suffix)).unlink()
            except FileNotFoundError:
                pass
        print("cleared previous results")

    # ---- 1. broker ---------------------------------------------------------
    if broker.port_is_open(config.MQTT_HOST, config.MQTT_PORT):
        print(f"[1/4] MQTT broker already running on {config.MQTT_HOST}:{config.MQTT_PORT} - using it")
        srv = None
    else:
        srv = broker.serve(config.MQTT_HOST, config.MQTT_PORT, background=True)
        print(f"[1/4] started built-in MQTT broker on {config.MQTT_HOST}:{config.MQTT_PORT}")
    time.sleep(0.5)

    # ---- 2. inference ------------------------------------------------------
    from gridtrace.infer import Pipeline
    from gridtrace.mqtt_util import make_client
    from gridtrace.storage import Store
    import json as _json

    store = Store()
    infer_client = make_client("gridtrace-infer")
    pipe = Pipeline(store, infer_client)
    infer_client.on_message = lambda _c, _u, m: _safe_handle(pipe, m)
    infer_client.connect(config.MQTT_HOST, config.MQTT_PORT, keepalive=30)
    infer_client.subscribe(config.TOPIC_ALL, qos=0)
    infer_client.loop_start()
    print(f"[2/4] inference online - models trained {pipe.bundle['created']} "
          f"on {pipe.bundle['n_train']:,} normal samples")

    # ---- 3. dashboard ------------------------------------------------------
    if not a.no_dashboard:
        from gridtrace import dashboard
        dashboard.store = store
        threading.Thread(
            target=lambda: dashboard.app.run(host=config.DASH_HOST, port=config.DASH_PORT,
                                             debug=False, threaded=True, use_reloader=False),
            daemon=True).start()
        url = f"http://{config.DASH_HOST}:{config.DASH_PORT}"
        print(f"[3/4] dashboard -> {url}")
        if not a.no_browser:
            threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    else:
        print("[3/4] dashboard disabled")

    # ---- 4. publisher ------------------------------------------------------
    print("[4/4] publishing telemetry - Ctrl-C to stop\n")
    from gridtrace.publisher import main as pub_main
    argv2 = ["--interval", str(a.interval)]
    argv2 += ["--mode", str(a.mode)] if a.mode is not None else ["--script", "demo"]

    try:
        pub_main(argv2)
    except KeyboardInterrupt:
        pass
    finally:
        s = pipe.stats
        print(f"\nsummary: received {s['received']}  scored {s['scored']}  "
              f"alerts {s['alerts']}  rejected {s['rejected']}")
        infer_client.loop_stop()
        infer_client.disconnect()
        store.close()
        if srv:
            srv.shutdown()
    return 0


def _safe_handle(pipe, msg):
    import json
    try:
        pipe.handle(json.loads(msg.payload.decode("utf-8", errors="replace")))
    except json.JSONDecodeError:
        pipe.stats["rejected"] += 1
    except Exception:
        logging.getLogger("gridtrace").exception("handler error")


if __name__ == "__main__":
    sys.exit(main())
