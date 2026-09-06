"""
Feature engineering for GridTrace AI.
=====================================

One `FeatureEngine` instance keeps per-consumer rolling state and turns each
incoming telemetry frame into a fixed-length feature vector.

The design decision that matters here: **the features are consumer-relative.**
C004 (commercial, ~3.6 kW) and C001 (low residential, ~0.45 kW) are not
comparable in raw watts, so a single global model trained on raw power would
just learn "C004 is big" and flag C001's evening peak as an outlier. Instead the
model sees how far each consumer is from *its own* baseline - a z-score against
that consumer's training statistics, and a ratio against that consumer's own
profile for this hour of day. One model then generalises across all five meters,
which is also how you would scale this to 50,000 of them.

The engine is used identically in training and inference, so there is no
train/serve skew.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any

from gridtrace import config


def _f(value: Any, default: float = 0.0) -> float:
    """Coerce anything the wire hands us into a finite float. Never raises."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(x) or math.isinf(x):
        return default
    return x


class Baselines:
    """Per-consumer statistics learned from normal data, used to relativise features."""

    def __init__(self):
        self.mean_kw: dict[str, float] = {}
        self.std_kw: dict[str, float] = {}
        self.hourly_kw: dict[tuple[str, int], float] = {}

    def fit(self, rows: list[dict]) -> "Baselines":
        acc: dict[str, list[float]] = defaultdict(list)
        hourly: dict[tuple[str, int], list[float]] = defaultdict(list)
        for r in rows:
            cid = r.get("consumer_id")
            if not cid:
                continue
            kw = _f(r.get("power")) / 1000.0
            acc[cid].append(kw)
            hourly[(cid, _hour_of(r))].append(kw)
        for cid, vals in acc.items():
            n = len(vals) or 1
            m = sum(vals) / n
            var = sum((v - m) ** 2 for v in vals) / n
            self.mean_kw[cid] = m
            self.std_kw[cid] = max(math.sqrt(var), 0.02)   # floor: a flat meter is not infinitely sensitive
        for key, vals in hourly.items():
            self.hourly_kw[key] = sum(vals) / len(vals)
        return self

    def mean_for(self, cid: str) -> float:
        return self.mean_kw.get(cid, 1.0) or 1.0

    def std_for(self, cid: str) -> float:
        return self.std_kw.get(cid, 0.25) or 0.25

    def hourly_for(self, cid: str, hour: int) -> float:
        return self.hourly_kw.get((cid, hour)) or self.mean_for(cid)

    def to_dict(self) -> dict:
        return {
            "mean_kw": self.mean_kw,
            "std_kw": self.std_kw,
            "hourly_kw": {f"{c}|{h}": v for (c, h), v in self.hourly_kw.items()},
        }

    @staticmethod
    def from_dict(d: dict) -> "Baselines":
        b = Baselines()
        b.mean_kw = dict(d.get("mean_kw", {}))
        b.std_kw = dict(d.get("std_kw", {}))
        b.hourly_kw = {}
        for k, v in d.get("hourly_kw", {}).items():
            c, _, h = k.partition("|")
            b.hourly_kw[(c, int(h))] = v
        return b


def _hour_of(rec: dict) -> int:
    ts = rec.get("timestamp") or ""
    try:
        return int(str(ts)[11:13])
    except (ValueError, IndexError):
        return 0


class FeatureEngine:
    """Streaming feature builder. Feed it frames in order; it holds the history."""

    def __init__(self, baselines: Baselines | None = None):
        self.baselines = baselines or Baselines()
        self.hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=config.ROLL_WINDOW))
        self.last_energy: dict[str, float] = {}
        self.feeder_gap_pct: float = config.TECHNICAL_LOSS_PCT

    # -------------------------------------------------------------- network
    def update_feeder(self, feeder_rec: dict) -> None:
        """Take the instantaneous gap, not the cumulative one.

        The cumulative `unaccounted_pct` is diluted by every hour of history, so a
        meter running for a week barely reacts to a new gap. `unaccounted_power_pct`
        is history-free and is the signal the model should see.
        """
        gap = feeder_rec.get("unaccounted_power_pct")
        if gap is None:                                     # older firmware: fall back
            gap = feeder_rec.get("unaccounted_pct")
        self.feeder_gap_pct = _f(gap, config.TECHNICAL_LOSS_PCT)

    def network_excess(self) -> float:
        """Feeder gap above the technical-loss floor, in percentage points (>= 0)."""
        return max(0.0, self.feeder_gap_pct - config.TECHNICAL_LOSS_PCT)

    # -------------------------------------------------------------- consumer
    def transform(self, rec: dict) -> tuple[list[float], dict] | None:
        """Return (feature_vector, context) or None if the record is unusable."""
        cid = rec.get("consumer_id")
        if not cid or rec.get("record_type") != "consumer":
            return None

        voltage = _f(rec.get("voltage"))
        current = _f(rec.get("current"))
        power_kw = _f(rec.get("power")) / 1000.0
        pf = _f(rec.get("power_factor"))
        energy = _f(rec.get("energy"))

        # A meter reporting nothing sane is a data-quality event, not an anomaly.
        if voltage <= 0 or pf <= 0:
            return None

        prev_e = self.last_energy.get(cid)
        # Energy is a monotonic accumulator; a decrease means a reset, not negative use.
        energy_delta = max(0.0, energy - prev_e) if prev_e is not None else 0.0
        self.last_energy[cid] = energy

        h = self.hist[cid]
        h.append(power_kw)
        if len(h) < config.MIN_HISTORY:
            return None                                     # not enough history to judge

        roll_mean = sum(h) / len(h)
        roll_var = sum((v - roll_mean) ** 2 for v in h) / len(h)
        roll_std = math.sqrt(roll_var)

        base_mean = self.baselines.mean_for(cid)
        base_std = self.baselines.std_for(cid)
        hour_mean = self.baselines.hourly_for(cid, _hour_of(rec))

        dev = (power_kw - base_mean) / base_std                     # consumer-specific z
        roll_ratio = roll_mean / base_mean if base_mean else 1.0
        roll_cv = roll_std / roll_mean if roll_mean > 1e-6 else 0.0
        hour_ratio = power_kw / hour_mean if hour_mean > 1e-6 else 1.0

        vec = [
            voltage,
            current,
            power_kw,
            pf,
            energy_delta,
            dev,
            roll_ratio,
            roll_cv,
            hour_ratio,
        ]
        vec = [_f(v) for v in vec]

        ctx = {
            "consumer_id": cid,
            "zone_id": rec.get("zone_id", "?"),
            "feeder_id": rec.get("feeder_id", config.FEEDER_ID),
            "timestamp": rec.get("timestamp", ""),
            "tick": int(_f(rec.get("tick"))),
            "voltage": voltage,
            "current": current,
            "power_w": power_kw * 1000.0,
            "power_factor": pf,
            "energy_kwh": energy,
            "dev_from_own_baseline": dev,
            "hourly_profile_ratio": hour_ratio,
            "feeder_gap_pct": self.feeder_gap_pct,
            "network_excess_pct": self.network_excess(),
            # ground truth, carried for evaluation only - never fed to the models
            "gt_label": rec.get("label", "unknown"),
            "gt_injected": bool(rec.get("anomaly_injected", False)),
            "mode": rec.get("mode", "?"),
        }
        return vec, ctx

    def reset(self) -> None:
        self.hist.clear()
        self.last_energy.clear()
        self.feeder_gap_pct = config.TECHNICAL_LOSS_PCT


def frames_to_matrix(records: list[dict], baselines: Baselines):
    """Batch helper for training: replay records through the engine in order."""
    eng = FeatureEngine(baselines)
    X, ctxs = [], []
    for rec in records:
        rt = rec.get("record_type")
        if rt == "feeder":
            eng.update_feeder(rec)
        elif rt == "consumer":
            out = eng.transform(rec)
            if out:
                X.append(out[0])
                ctxs.append(out[1])
    return X, ctxs
