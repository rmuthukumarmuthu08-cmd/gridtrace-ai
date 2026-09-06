"""Local SQLite storage. No server, one file, safe for concurrent reader + writer."""
from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any

from gridtrace import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, tick INTEGER,
    consumer_id TEXT, zone_id TEXT, feeder_id TEXT,
    voltage REAL, current REAL, power_w REAL, power_factor REAL, energy_kwh REAL,
    iforest_score REAL, autoencoder_score REAL, hybrid_score REAL,
    risk_score REAL, risk_band TEXT,
    network_evidence REAL, persistence REAL, feeder_gap_pct REAL,
    reasons TEXT, gt_label TEXT, gt_injected INTEGER, mode TEXT
);
CREATE INDEX IF NOT EXISTS idx_readings_tick ON readings(tick);
CREATE INDEX IF NOT EXISTS idx_readings_consumer ON readings(consumer_id);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, tick INTEGER,
    level TEXT, feeder_id TEXT, zone_id TEXT, consumer_id TEXT,
    risk_score REAL, risk_band TEXT, unattributed INTEGER,
    summary TEXT, reasons TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_tick ON alerts(tick);
"""


class Store:
    def __init__(self, path=None):
        self.path = str(path or config.DB_PATH)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")   # reader (dashboard) never blocks writer
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def save_reading(self, ctx: dict, res: dict, reasons: list[str]) -> None:
        with self._lock:
            self.conn.execute(
                """INSERT INTO readings (ts,tick,consumer_id,zone_id,feeder_id,voltage,current,
                   power_w,power_factor,energy_kwh,iforest_score,autoencoder_score,hybrid_score,
                   risk_score,risk_band,network_evidence,persistence,feeder_gap_pct,reasons,
                   gt_label,gt_injected,mode)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (ctx.get("timestamp"), ctx.get("tick"), ctx.get("consumer_id"),
                 ctx.get("zone_id"), ctx.get("feeder_id"), ctx.get("voltage"),
                 ctx.get("current"), ctx.get("power_w"), ctx.get("power_factor"),
                 ctx.get("energy_kwh"), res.get("iforest_score"), res.get("autoencoder_score"),
                 res.get("hybrid_score"), res.get("risk_score"), res.get("risk_band"),
                 res.get("network_evidence"), res.get("persistence"), ctx.get("feeder_gap_pct"),
                 json.dumps(reasons), ctx.get("gt_label"), int(bool(ctx.get("gt_injected"))),
                 ctx.get("mode")))
            self.conn.commit()

    def save_alert(self, loc: dict, tick: int, ts: str, reasons: list[str]) -> None:
        with self._lock:
            self.conn.execute(
                """INSERT INTO alerts (ts,tick,level,feeder_id,zone_id,consumer_id,
                   risk_score,risk_band,unattributed,summary,reasons)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (ts, tick, loc.get("level"), loc.get("feeder_id"), loc.get("zone_id"),
                 loc.get("consumer_id"), loc.get("risk_score"), loc.get("risk_band"),
                 int(bool(loc.get("unattributed"))), loc.get("summary"), json.dumps(reasons)))
            self.conn.commit()

    # -------------------------------------------------------------- queries
    def _rows(self, sql: str, args=()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def latest_per_consumer(self) -> list[dict]:
        return self._rows("""
            SELECT r.* FROM readings r
            JOIN (SELECT consumer_id, MAX(id) AS mid FROM readings GROUP BY consumer_id) t
              ON r.id = t.mid
            ORDER BY r.consumer_id""")

    def recent_alerts(self, limit=25) -> list[dict]:
        return self._rows("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))

    def risk_series(self, limit=180) -> list[dict]:
        return self._rows("""
            SELECT tick, MAX(risk_score) AS risk, MAX(feeder_gap_pct) AS gap
            FROM readings GROUP BY tick ORDER BY tick DESC LIMIT ?""", (limit,))[::-1]

    def counts(self) -> dict:
        rows = self._rows("SELECT risk_band, COUNT(*) c FROM readings GROUP BY risk_band")
        out = {b: 0 for _, _, b in config.RISK_BANDS}
        for r in rows:
            out[r["risk_band"]] = r["c"]
        total = self._rows("SELECT COUNT(*) c FROM readings")[0]["c"]
        return {"bands": out, "total": total}

    def close(self):
        with self._lock:
            self.conn.close()
