"""
Hybrid intelligence: one score from two models, then a risk score.
==================================================================

THE PROBLEM WITH NAIVELY COMBINING THEM
Isolation Forest returns a path-length score around [-0.5, +0.5]; the autoencoder
returns a reconstruction MSE in [0, inf) whose scale depends entirely on how well
training converged. Averaging those two numbers directly is meaningless - whichever
happens to have the larger numeric range dominates, and the "hybrid" is really just
one model wearing two names.

THE FIX: CALIBRATE BOTH TO THE SAME UNIT FIRST
At training time we score the *normal* data with each model and keep those score
distributions. At inference, a raw score is converted to its **empirical percentile
within the normal distribution** - "what fraction of known-normal behaviour is less
anomalous than this?". Both models now emit a number in [0, 1] with an identical,
interpretable meaning, and neither can dominate by scale accident.

COMBINING: PART AGREEMENT, PART SENSITIVITY
    s_model = W_MEAN * mean(s_if, s_ae) + (1 - W_MEAN) * max(s_if, s_ae)

The mean term rewards *agreement* - when both models independently find a point
unusual, that is much stronger evidence than one model alone, and averaging
suppresses single-model false positives. The max term preserves *sensitivity* -
the two models fail differently (Isolation Forest is good at coordinate-wise
outliers, the autoencoder at broken relationships between features), so a genuine
anomaly that only one of them can see must still survive. Pure mean is too
conservative, pure max too jumpy; the even split is the useful middle.

RISK SCORE
    risk = 100 * s_model  +  W_NETWORK * network_evidence  +  W_PERSISTENCE * persistence

The model term is the body of the score. The two additive terms are *supporting
evidence*, deliberately capped small (12 points each) so that no amount of
feeder imbalance alone can push a well-behaved consumer into HIGH RISK - network
imbalance is a hint about where to look, never a verdict about who did it.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from gridtrace import config


class Calibrator:
    """Maps a raw model score to its percentile within the normal-data distribution."""

    def __init__(self, normal_scores=None):
        self.q = np.asarray(sorted(normal_scores), dtype=float) if normal_scores is not None \
            else np.array([0.0, 1.0])

    def percentile(self, x):
        """Fraction of normal scores at or below x, in [0, 1]. Vectorised."""
        x = np.asarray(x, dtype=float)
        pos = np.searchsorted(self.q, x, side="right")
        return np.clip(pos / max(len(self.q), 1), 0.0, 1.0)

    def to_dict(self):
        # Store 512 quantiles rather than every point: same behaviour, small bundle.
        n = min(512, len(self.q))
        idx = np.linspace(0, len(self.q) - 1, n).astype(int)
        return {"q": self.q[idx].tolist()}

    @staticmethod
    def from_dict(d):
        return Calibrator(d["q"])


class HybridScorer:
    """Turns one feature vector into a calibrated hybrid score, risk score and band."""

    def __init__(self, iforest, scaler, autoencoder, cal_if: Calibrator, cal_ae: Calibrator):
        self.iforest = iforest
        self.scaler = scaler
        self.ae = autoencoder
        self.cal_if = cal_if
        self.cal_ae = cal_ae
        self._recent: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=config.PERSISTENCE_WINDOW))

    # ------------------------------------------------------------- raw scores
    def raw_scores(self, X):
        """(iforest_score, ae_error, per_feature_error) for a batch of raw features."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Xs = self.scaler.transform(X)
        # sklearn's score_samples: higher = more normal. Negate so higher = more anomalous.
        s_if = -self.iforest.score_samples(Xs)
        err = self.ae.reconstruction_error(Xs)
        per_feat = self.ae.per_feature_error(Xs)
        return s_if, err, per_feat

    # ------------------------------------------------------------- hybrid
    @staticmethod
    def severity(p):
        """Percentile -> severity in [0, 1].

        A percentile alone is the wrong scale for risk: on normal data percentiles
        are uniform, so half of perfectly healthy readings sit above 0.5 and would
        score 50/100. What matters is how far into the *tail* a reading is, so we
        use the exceedance probability on a log scale:

            severity = -log10(1 - p) / SEVERITY_NINES

        Each extra "nine" of rarity adds a fixed amount of risk. With three nines:
        median normal -> 0.10, 90th pct -> 0.33, 99th -> 0.67, 99.9th -> 1.0.
        Typical normal behaviour therefore lands in the NORMAL band, and only
        genuinely rare readings approach the top.
        """
        p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
        tail = np.maximum(1.0 - p, 1e-4)
        return np.clip(-np.log10(tail) / config.SEVERITY_NINES, 0.0, 1.0)

    def score_one(self, x, ctx: dict) -> dict:
        s_if_raw, ae_raw, per_feat = self.raw_scores([x])
        p_if = float(self.cal_if.percentile(s_if_raw)[0])
        p_ae = float(self.cal_ae.percentile(ae_raw)[0])
        s_if = float(self.severity(p_if))
        s_ae = float(self.severity(p_ae))

        s_model = config.W_MEAN * (0.5 * (s_if + s_ae)) + (1 - config.W_MEAN) * max(s_if, s_ae)

        # --- supporting evidence 1: network-level imbalance -------------------
        # Excess over the technical-loss floor, saturating at 12 points of excess.
        excess = float(ctx.get("network_excess_pct", 0.0))
        network_evidence = min(1.0, excess / 12.0)

        # --- supporting evidence 2: is it sustained? --------------------------
        cid = ctx.get("consumer_id", "?")
        hist = self._recent[cid]
        hist.append(1 if s_model >= 0.90 else 0)
        persistence = sum(hist) / hist.maxlen if hist.maxlen else 0.0

        risk = 100.0 * s_model + config.W_NETWORK * network_evidence \
            + config.W_PERSISTENCE * persistence
        risk = float(np.clip(risk, 0.0, 100.0))

        return {
            "iforest_raw": float(s_if_raw[0]),
            "ae_error_raw": float(ae_raw[0]),
            "iforest_pct": p_if,
            "autoencoder_pct": p_ae,
            "iforest_score": s_if,
            "autoencoder_score": s_ae,
            "hybrid_score": float(s_model),
            "network_evidence": network_evidence,
            "persistence": float(persistence),
            "risk_score": round(risk, 1),
            "risk_band": config.risk_band(risk),
            "per_feature_error": per_feat[0].tolist(),
        }

    def reset_persistence(self):
        self._recent.clear()
