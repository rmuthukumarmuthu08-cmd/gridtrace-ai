#!/usr/bin/env python3
"""
Telemetry source for GridTrace AI (port of the ESP32/Wokwi firmware core).
=============================================================

A faithful port of the ESP32 firmware's simulation core (src/main.cpp), so you
can produce as much labelled telemetry as you want WITHOUT Wokwi, a board, or a
working Serial Monitor. Same model, same fixed-seed RNG, same NDJSON schema.

    python simulate.py                          # 4 modes x 300 ticks -> stdout
    python simulate.py --ticks 900 --out data.ndjson
    python simulate.py --mode NETWORK_IMBALANCE --ticks 600
    python simulate.py --seed 12345             # independent dataset
    python simulate.py --load 1.3               # heavier grid demand

Then:  python validate.py data.ndjson

NOTE ON FIDELITY: the firmware computes in 32-bit float, this runs in 64-bit.
Individual readings differ in about the 7th significant digit; every aggregate
(mean power per consumer, unaccounted %, mode separation) matches. Use this for
dataset generation and model development; use the firmware for the hardware demo.

This is a MEASUREMENT simulator. It perturbs numbers in software only. It
contains no mechanism for interfering with any real meter or supply, and an
energy gap in this data is NOT evidence of theft - it is equally consistent with
technical loss, a faulty meter, or a metering/data gap.
"""

import argparse
import json
import math
import sys
import time

# ----------------------------------------------------------------- constants
SIM_SECONDS_PER_TICK = 60
SIM_EPOCH_START = 1788588000        # 2026-09-05T06:00:00Z
V_NOMINAL = 233.0
V_DROOP_PER_KW = 0.42
V_MIN, V_MAX = 200.0, 250.0
TECH_LOSS_FRAC = 0.032
DEV_THRESHOLD = 0.35

MODES = ["NORMAL", "ABNORMAL_CONSUMPTION", "PERSISTENT_ANOMALY", "NETWORK_IMBALANCE"]
LABELS = ["normal", "abnormal_consumption", "persistent_deviation", "network_imbalance"]

# id, zone, feeder, load_type, profile, base_kw, pf_nominal, zone_drop_v
CONSUMERS = [
    ("C001", "Z01", "F01", "residential_low",    "res_low",    0.45, 0.97, 1.4),
    ("C002", "Z01", "F01", "residential_medium", "res_med",    1.10, 0.95, 1.6),
    ("C003", "Z01", "F01", "residential_high",   "res_high",   2.20, 0.93, 1.9),
    ("C004", "Z02", "F01", "commercial",         "commercial", 3.60, 0.89, 3.2),
    ("C005", "Z02", "F01", "variable_load",      "variable",   1.60, 0.91, 3.5),
]


class LCG:
    """Same linear congruential generator as the firmware, so runs reproduce."""

    def __init__(self, seed=0xC0FFEE):
        self.s = seed & 0xFFFFFFFF

    def next(self):
        self.s = (self.s * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.s

    def r01(self):
        return (self.next() >> 8) / 16777216.0

    def sym(self):
        return self.r01() * 2.0 - 1.0


def gauss_pk(x, mu, sigma):
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z)


def clampf(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def profile_shape(p, h):
    """Dimensionless load multiplier (~0.2 .. ~1.6) for hour-of-day h."""
    if p == "res_low":
        return 0.30 + 0.45 * gauss_pk(h, 7.5, 1.4) + 0.85 * gauss_pk(h, 20.0, 2.0)
    if p == "res_med":
        return 0.35 + 0.60 * gauss_pk(h, 8.0, 1.6) + 1.00 * gauss_pk(h, 20.5, 2.2)
    if p == "res_high":
        return (0.40 + 0.70 * gauss_pk(h, 7.0, 1.5) + 1.10 * gauss_pk(h, 21.0, 2.4)
                + 0.30 * gauss_pk(h, 14.0, 2.0))
    if p == "commercial":
        on = 1.0 / (1.0 + math.exp(-(h - 9.0) * 2.2))
        off = 1.0 / (1.0 + math.exp(-(h - 18.5) * 2.2))
        return 0.20 + 1.10 * (on - off)
    if p == "variable":
        return (0.45 + 0.55 * (0.5 + 0.5 * math.sin(h * 0.9))
                + 0.35 * gauss_pk(h, 12.0, 1.2))
    return 1.0


def iso(epoch):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


class Simulator:
    def __init__(self, seed=0xC0FFEE, load=1.0):
        self.rng = LCG(seed)
        self.grid_scale = load
        self.tick = 0
        self.epoch = SIM_EPOCH_START
        self.walk = [0.0] * 5
        self.true_energy = [0.0] * 5
        self.meter_energy = [0.0] * 5
        self.ewma = [0.0] * 5
        self.feeder_energy = 0.0
        self.feeder_kw = 0.0

    def step(self, mode_idx):
        """Advance one tick. Returns the list of records for this frame."""
        h = (self.epoch % 86400) / 3600.0
        dt_hour = SIM_SECONDS_PER_TICK / 3600.0

        # pass 1: true delivered power
        true_kw, meter_factor, injected = [], [1.0] * 5, [False] * 5
        for i, c in enumerate(CONSUMERS):
            self.walk[i] = clampf(self.walk[i] * 0.92 + self.rng.sym() * 0.030, -0.15, 0.15)
            true_kw.append(c[5] * profile_shape(c[4], h) * self.grid_scale * (1.0 + self.walk[i]))

        # anomaly injection - software only, measurement level
        extra_unacc = 0.0
        if mode_idx == 1:                                   # ABNORMAL_CONSUMPTION
            if self.tick % 40 < 15:
                true_kw[2] *= 2.40
                injected[2] = True
            if self.tick % 37 < 5:
                meter_factor[4] = 0.25
                injected[4] = True
        elif mode_idx == 2:                                 # PERSISTENT_ANOMALY
            meter_factor[1] = 0.45
            injected[1] = True
            if h < 5.0 or h > 23.0:
                true_kw[3] += 2.20
                injected[3] = True

        total_true = sum(true_kw)
        if mode_idx == 3:                                   # NETWORK_IMBALANCE
            extra_unacc = (0.13 + 0.04 * math.sin(self.tick * 0.05)) * total_true

        # pass 2: bus voltage sags with feeder loading
        bus_v = clampf(V_NOMINAL - V_DROOP_PER_KW * total_true + self.rng.sym() * 0.35,
                       V_MIN, V_MAX)

        # pass 3: per-consumer electrical quantities
        recs, sum_meter_e = [], 0.0
        zone_kw, zone_e = {"Z01": 0.0, "Z02": 0.0}, {"Z01": 0.0, "Z02": 0.0}
        ts, mode, label = iso(self.epoch), MODES[mode_idx], LABELS[mode_idx]

        for i, c in enumerate(CONSUMERS):
            v = clampf(bus_v - c[7] + self.rng.sym() * 0.25, V_MIN, V_MAX)
            pf = clampf(c[6] + self.rng.sym() * 0.015, 0.75, 1.00)
            meas_kw = true_kw[i] * meter_factor[i]
            amps = (meas_kw * 1000.0) / (v * pf)            # keeps P = V*I*PF exact

            self.true_energy[i] += true_kw[i] * dt_hour
            self.meter_energy[i] += meas_kw * dt_hour

            if self.tick == 0:
                self.ewma[i] = meas_kw
            dev = (meas_kw - self.ewma[i]) / self.ewma[i] if self.ewma[i] > 0.05 else 0.0
            self.ewma[i] = 0.97 * self.ewma[i] + 0.03 * meas_kw
            status = "SURGE" if dev > DEV_THRESHOLD else ("DIP" if dev < -DEV_THRESHOLD else "OK")

            sum_meter_e += self.meter_energy[i]
            zone_kw[c[1]] += meas_kw
            zone_e[c[1]] += self.meter_energy[i]

            recs.append({
                "record_type": "consumer", "schema": "els.v1", "timestamp": ts,
                "uptime_ms": self.tick * 1000, "tick": self.tick,
                "consumer_id": c[0], "zone_id": c[1], "feeder_id": c[2], "load_type": c[3],
                "voltage": round(v, 2), "current": round(amps, 3),
                "power": round(meas_kw * 1000.0, 1), "power_factor": round(pf, 3),
                "energy": round(self.meter_energy[i], 4),
                "status": status, "meter_status": "ONLINE",
                "mode": mode, "label": label, "anomaly_injected": injected[i],
            })

        for z in ("Z01", "Z02"):
            recs.append({
                "record_type": "zone", "schema": "els.v1", "timestamp": ts,
                "uptime_ms": self.tick * 1000, "tick": self.tick,
                "zone_id": z, "feeder_id": "F01",
                "metered_power": round(zone_kw[z] * 1000.0, 1),
                "metered_energy": round(zone_e[z], 4),
                "mode": mode, "label": label,
            })

        # pass 4: feeder-head energy balance
        self.feeder_kw = total_true * (1.0 + TECH_LOSS_FRAC) + extra_unacc
        self.feeder_energy += self.feeder_kw * dt_hour
        unacc = self.feeder_energy - sum_meter_e
        unacc_pct = (100.0 * unacc / self.feeder_energy) if self.feeder_energy > 1e-4 else 0.0

        # INSTANTANEOUS gap. The cumulative figure above is diluted by every hour
        # of history, so a long-running meter barely reacts to a new gap. This one
        # is history-free and is the signal an anomaly detector actually wants.
        sum_meter_kw = zone_kw["Z01"] + zone_kw["Z02"]
        unacc_kw = self.feeder_kw - sum_meter_kw
        unacc_kw_pct = (100.0 * unacc_kw / self.feeder_kw) if self.feeder_kw > 1e-3 else 0.0

        recs.append({
            "record_type": "feeder", "schema": "els.v1", "timestamp": ts,
            "uptime_ms": self.tick * 1000, "tick": self.tick,
            "feeder_id": "F01", "consumers": 5,
            "input_power": round(self.feeder_kw * 1000.0, 1),
            "input_energy": round(self.feeder_energy, 4),
            "sum_meter_energy": round(sum_meter_e, 4),
            "unaccounted_energy": round(unacc, 4),
            "unaccounted_pct": round(unacc_pct, 2),
            "unaccounted_power": round(unacc_kw * 1000.0, 1),
            "unaccounted_power_pct": round(unacc_kw_pct, 2),
            "technical_loss_pct_expected": round(100.0 * TECH_LOSS_FRAC / (1.0 + TECH_LOSS_FRAC), 2),
            "grid_load_scale": round(self.grid_scale, 2),
            "mode": mode, "label": label,
            "note": "SIMULATED NETWORK IMBALANCE - unaccounted energy is not proof of theft",
        })

        self.epoch += SIM_SECONDS_PER_TICK
        self.tick += 1
        return recs


# The CLI lives in publisher.py / train.py; this module is import-only.
# `Simulator(seed, load).step(mode_idx)` returns one frame of records:
#   5 consumer + 2 zone + 1 feeder, exactly the schema the ESP32 firmware emits.
