"""
Localization and explanation.
=============================

Localization walks the hierarchy Network -> Feeder -> Zone -> Consumer and answers
"where should an inspector actually go?" for the current frame.

The case that makes this worth doing: when the feeder shows a large energy gap but
no individual consumer looks unusual, the honest answer is *not* to accuse the
highest-scoring consumer. It is to localize at FEEDER level and say the loss is
unattributed - which is exactly what a distribution-loss or unmetered-connection
scenario looks like. The function below distinguishes those two cases explicitly.

Explanations are derived from the values that actually produced the score - the
calibrated model percentiles, the per-feature reconstruction error, the consumer's
own deviation, persistence, and the network gap. Nothing is hard-coded per mode.
"""
from __future__ import annotations

from gridtrace import config

# Human wording for each feature the autoencoder can fail to reconstruct.
FEATURE_PHRASE = {
    "voltage": "supply voltage outside its usual band",
    "current": "current draw unlike this meter's normal range",
    "power_kw": "power level far from this consumer's baseline",
    "power_factor": "power factor departing from its usual value",
    "energy_delta_kwh": "energy registered per interval inconsistent with measured power",
    "dev_from_own_baseline": "significant deviation from this consumer's own normal behaviour",
    "roll_mean_ratio": "sustained shift in the rolling consumption average",
    "roll_cv": "unusual volatility in consumption",
    "hourly_profile_ratio": "consumption unusual for this time of day",
    "feeder_gap_pct": "feeder-level energy imbalance coinciding with this reading",
}


def explain(result: dict, ctx: dict) -> list[str]:
    """Ordered, human-readable contributing factors for a scored consumer reading."""
    reasons: list[str] = []

    # Thresholds are on the calibrated PERCENTILE (where this reading sits inside
    # the distribution of known-normal behaviour), not the severity, because
    # "rarer than 99% of normal" is the statement a human can actually check.
    p_ae = result.get("autoencoder_pct", 0.0)
    p_if = result.get("iforest_pct", 0.0)

    if p_ae >= 0.999:
        reasons.append("Very high autoencoder reconstruction error - the pattern does not "
                       "match any normal behaviour the model learned")
    elif p_ae >= 0.95:
        reasons.append(f"Elevated autoencoder reconstruction error "
                       f"(rarer than {p_ae:.1%} of normal readings)")

    if p_if >= 0.999:
        reasons.append("Isolation Forest isolates this reading from the normal population")
    elif p_if >= 0.95:
        reasons.append(f"Elevated Isolation Forest anomaly score "
                       f"(rarer than {p_if:.1%} of normal readings)")

    if p_if >= 0.95 and p_ae >= 0.95:
        reasons.append("Both models agree independently, which raises confidence")

    dev = abs(float(ctx.get("dev_from_own_baseline", 0.0)))
    if dev >= 3.0:
        reasons.append(f"Consumption is {dev:.1f} standard deviations from this "
                       f"consumer's own baseline")
    elif dev >= 2.0:
        reasons.append(f"Consumption is {dev:.1f} sigma from this consumer's own baseline")

    hr = float(ctx.get("hourly_profile_ratio", 1.0))
    if hr >= 1.8:
        reasons.append(f"Draw is {hr:.1f}x this consumer's typical level for this hour")
    elif 0 < hr <= 0.45:
        reasons.append(f"Registered consumption is only {hr:.0%} of the typical level "
                       f"for this hour - possible under-registration")

    # top autoencoder feature failures
    pfe = result.get("per_feature_error") or []
    if pfe:
        ranked = sorted(zip(config.FEATURE_NAMES, pfe), key=lambda kv: -kv[1])
        total = sum(pfe) or 1.0
        for name, val in ranked[:2]:
            if val / total >= 0.25:
                reasons.append(FEATURE_PHRASE.get(name, name))

    if result["persistence"] >= 0.5:
        reasons.append(f"Pattern is persistent - flagged in "
                       f"{result['persistence']:.0%} of the last "
                       f"{config.PERSISTENCE_WINDOW} intervals")

    if result["network_evidence"] >= 0.3:
        reasons.append(f"Feeder-level imbalance of "
                       f"{ctx.get('feeder_gap_pct', 0):.1f}% is running above the "
                       f"{config.TECHNICAL_LOSS_PCT:.1f}% technical-loss baseline")

    if not reasons:
        reasons.append("No individual factor stands out; score reflects mild combined deviation")
    if len(reasons) < 2:
        reasons.append(
            f"Combined hybrid anomaly score {result['hybrid_score']:.2f} "
            f"(Isolation Forest {result['iforest_score']:.2f}, "
            f"autoencoder {result['autoencoder_score']:.2f})")
    return reasons


def localize(scored: list[dict], feeder_gap_pct: float) -> dict:
    """
    Roll consumer results up the hierarchy and pick the level to investigate.

    `scored` is a list of dicts each carrying consumer_id / zone_id / feeder_id /
    risk_score. Returns the localization verdict for this frame.
    """
    net_excess = max(0.0, feeder_gap_pct - config.TECHNICAL_LOSS_PCT)

    if not scored:
        return {
            "level": "FEEDER", "feeder_id": config.FEEDER_ID, "zone_id": None,
            "consumer_id": None, "risk_score": 0.0, "risk_band": "NORMAL",
            "feeder_gap_pct": round(feeder_gap_pct, 2),
            "unattributed": False,
            "summary": "Warming up - not enough history to score consumers yet.",
        }

    ranked = sorted(scored, key=lambda r: -r["risk_score"])
    top = ranked[0]

    # zone roll-up: a zone is as risky as its worst meter, nudged by how many are hot
    zones: dict[str, list[dict]] = {}
    for r in scored:
        zones.setdefault(r["zone_id"], []).append(r)
    zone_scores = {
        z: max(x["risk_score"] for x in rs) + 3.0 * (sum(1 for x in rs if x["risk_score"] >= 61) - 1)
        for z, rs in zones.items()
    }
    worst_zone = max(zone_scores, key=zone_scores.get) if zone_scores else None

    # The key judgement: a big feeder gap that no consumer explains is a FEEDER-level
    # finding. Blaming the highest-scoring meter here would be exactly the false
    # accusation this system is supposed to avoid.
    unattributed = net_excess >= 3.0 and top["risk_score"] < 61

    if unattributed:
        risk = min(100.0, 45.0 + 3.0 * net_excess)
        return {
            "level": "FEEDER",
            "feeder_id": top["feeder_id"],
            "zone_id": None,
            "consumer_id": None,
            "risk_score": round(risk, 1),
            "risk_band": config.risk_band(risk),
            "feeder_gap_pct": round(feeder_gap_pct, 2),
            "unattributed": True,
            "summary": (f"Feeder {top['feeder_id']} is losing {net_excess:.1f} percentage points "
                        f"above its technical-loss baseline, but no individual meter accounts "
                        f"for it. Unattributed loss - inspect the feeder segment, check for "
                        f"unmetered connections or a metering gap before suspecting any consumer."),
        }

    return {
        "level": "CONSUMER" if top["risk_score"] >= 31 else "NETWORK",
        "feeder_id": top["feeder_id"],
        "zone_id": top["zone_id"] if top["risk_score"] >= 31 else worst_zone,
        "consumer_id": top["consumer_id"] if top["risk_score"] >= 31 else None,
        "risk_score": top["risk_score"],
        "risk_band": top["risk_band"],
        "feeder_gap_pct": round(feeder_gap_pct, 2),
        "unattributed": False,
        "summary": (
            f"Highest risk is {top['consumer_id']} in zone {top['zone_id']} on feeder "
            f"{top['feeder_id']} at {top['risk_score']:.0f}/100 ({top['risk_band']})."
            if top["risk_score"] >= 31 else
            "All meters within normal behaviour; feeder balance consistent with technical loss."
        ),
    }
