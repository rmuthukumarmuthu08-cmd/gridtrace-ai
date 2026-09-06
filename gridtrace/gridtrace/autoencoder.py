"""
Undercomplete autoencoder, implemented directly in NumPy.
=========================================================

Why not TensorFlow/PyTorch: this has to install and run on any laptop for a
prototype demo. A 10-7-3-7-10 autoencoder is ~150 parameters; pulling in a 500 MB
deep-learning runtime to train it would be the wrong trade. This is a real neural
network - dense layers, tanh activations, Adam, mini-batches, early stopping on a
held-out split - it just has no framework around it.

How it detects anomalies: it is trained *only on normal data* to reconstruct its
own input through a 3-unit bottleneck. To do that it has to learn the structure of
normal consumption. Feed it a pattern that structure doesn't cover and the
reconstruction degrades - the per-sample mean squared error is the anomaly score.
Because the bottleneck also tells us *which* inputs it failed to reproduce, the
per-feature error is what the explanation layer uses.
"""
from __future__ import annotations

import numpy as np


class Autoencoder:
    def __init__(self, n_in: int, hidden=(7, 3), lr=0.004, seed=42):
        self.n_in = int(n_in)
        self.hidden = tuple(int(h) for h in hidden)
        self.lr = float(lr)
        self.seed = int(seed)
        dims = [self.n_in, *self.hidden, *reversed(self.hidden[:-1]), self.n_in]
        self.dims = dims
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i in range(len(dims) - 1):
            # Xavier/Glorot: keeps activation variance stable through a tanh stack
            limit = np.sqrt(6.0 / (dims[i] + dims[i + 1]))
            self.W.append(rng.uniform(-limit, limit, (dims[i], dims[i + 1])))
            self.b.append(np.zeros(dims[i + 1]))
        self._reset_adam()
        self.history: dict[str, list[float]] = {"train": [], "val": []}

    def _reset_adam(self):
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    # ------------------------------------------------------------------ forward
    def _forward(self, X):
        acts = [X]
        a = X
        last = len(self.W) - 1
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = a @ w + b
            a = z if i == last else np.tanh(z)   # linear output: inputs are standardised, not bounded
            acts.append(a)
        return acts

    def predict(self, X):
        return self._forward(np.asarray(X, dtype=float))[-1]

    def reconstruction_error(self, X):
        """Per-sample mean squared error - this is the anomaly score."""
        X = np.asarray(X, dtype=float)
        return np.mean((self.predict(X) - X) ** 2, axis=1)

    def per_feature_error(self, X):
        """Squared error per input feature - drives the 'which signal broke' explanation."""
        X = np.asarray(X, dtype=float)
        return (self.predict(X) - X) ** 2

    # ------------------------------------------------------------------ training
    def _step(self, Xb):
        n = Xb.shape[0]
        acts = self._forward(Xb)
        out = acts[-1]
        delta = 2.0 * (out - Xb) / n            # dMSE/dout
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = acts[i].T @ delta
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (1.0 - acts[i] ** 2)   # tanh'
        # Adam
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i in range(len(self.W)):
            self.mW[i] = b1 * self.mW[i] + (1 - b1) * gW[i]
            self.vW[i] = b2 * self.vW[i] + (1 - b2) * (gW[i] ** 2)
            mh = self.mW[i] / (1 - b1 ** self.t)
            vh = self.vW[i] / (1 - b2 ** self.t)
            self.W[i] -= self.lr * mh / (np.sqrt(vh) + eps)

            self.mb[i] = b1 * self.mb[i] + (1 - b1) * gb[i]
            self.vb[i] = b2 * self.vb[i] + (1 - b2) * (gb[i] ** 2)
            mh = self.mb[i] / (1 - b1 ** self.t)
            vh = self.vb[i] / (1 - b2 ** self.t)
            self.b[i] -= self.lr * mh / (np.sqrt(vh) + eps)
        return float(np.mean((out - Xb) ** 2))

    def fit(self, X, epochs=300, batch=64, val_fraction=0.15, patience=30, verbose=True):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(X))
        n_val = max(1, int(len(X) * val_fraction))
        val, train = X[idx[:n_val]], X[idx[n_val:]]

        best, best_state, wait = np.inf, None, 0
        for ep in range(epochs):
            order = rng.permutation(len(train))
            losses = []
            for s in range(0, len(train), batch):
                xb = train[order[s:s + batch]]
                if len(xb) > 1:
                    losses.append(self._step(xb))
            tr = float(np.mean(losses)) if losses else np.nan
            vl = float(np.mean(self.reconstruction_error(val)))
            self.history["train"].append(tr)
            self.history["val"].append(vl)

            if vl < best - 1e-6:
                best, wait = vl, 0
                best_state = ([w.copy() for w in self.W], [b.copy() for b in self.b])
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"    early stop at epoch {ep + 1} (best val MSE {best:.6f})")
                    break
            if verbose and (ep + 1) % 50 == 0:
                print(f"    epoch {ep + 1:3d}  train {tr:.6f}  val {vl:.6f}")

        if best_state:
            self.W, self.b = best_state
        self.best_val = best
        return self

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return {
            "n_in": self.n_in, "hidden": self.hidden, "lr": self.lr, "seed": self.seed,
            "W": [w.tolist() for w in self.W], "b": [b.tolist() for b in self.b],
            "best_val": float(getattr(self, "best_val", 0.0)),
            "history": self.history,
        }

    @staticmethod
    def from_dict(d: dict) -> "Autoencoder":
        ae = Autoencoder(d["n_in"], tuple(d["hidden"]), d["lr"], d["seed"])
        ae.W = [np.array(w, dtype=float) for w in d["W"]]
        ae.b = [np.array(b, dtype=float) for b in d["b"]]
        ae.best_val = d.get("best_val", 0.0)
        ae.history = d.get("history", {"train": [], "val": []})
        return ae
