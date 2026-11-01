"""A small, dependency-free CMA-ES (Hansen 2016, "The CMA Evolution Strategy: A Tutorial"), in box bounds.

Used for the `numeric` gene. Works in a normalised [0, 1]^n space and maps to the bounds; state is JSON-serialisable so
it persists across generations in the orchestrator state.
"""
import math

import numpy as np


class CMAES:
    def __init__(self, x0, lo, hi, sigma=0.2, seed=0, _state=None):
        self.lo, self.hi = np.array(lo, float), np.array(hi, float)
        n = len(x0)
        self.n = n
        self.lam = 4 + int(3 * math.log(n))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1 / np.sum(self.w ** 2)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chin = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        if _state:
            self.m = np.array(_state["m"]); self.sigma = _state["sigma"]; self.C = np.array(_state["C"])
            self.pc = np.array(_state["pc"]); self.ps = np.array(_state["ps"]); self.gen = _state["gen"]
            self.rng = np.random.default_rng(_state["seed"] + _state["gen"] * 7919 + _state.get("asked", 0))
            self.seed = _state["seed"]; self.asked = _state.get("asked", 0); self.pending = _state.get("pending", [])
        else:
            self.m = self._norm(np.array(x0, float)); self.sigma = sigma; self.C = np.eye(n)
            self.pc = np.zeros(n); self.ps = np.zeros(n); self.gen = 0; self.seed = seed
            self.rng = np.random.default_rng(seed); self.asked = 0; self.pending = []

    def _norm(self, x):
        span = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        return (x - self.lo) / span

    def _denorm(self, z):
        return self.lo + np.clip(z, 0, 1) * (self.hi - self.lo)

    def _sample(self):
        B_D = np.linalg.cholesky(self.C + 1e-12 * np.eye(self.n))
        return self.m + self.sigma * B_D @ self.rng.standard_normal(self.n)

    def ask_one(self):
        z = self._sample()
        self.asked += 1
        self.pending.append(z.tolist())
        return self._denorm(z).tolist()

    def ask(self):
        return [self.ask_one() for _ in range(self.lam)]

    def tell(self, xs, fitness):
        """xs in original units, fitness higher-is-better. Standard rank-mu + rank-one update."""
        Z = np.array([np.clip(self._norm(np.array(x, float)), 0, 1) for x in xs])
        order = np.argsort(-np.array(fitness, float))[:self.mu]
        sel = Z[order]
        old = self.m
        self.m = self.w @ sel
        y = (self.m - old) / self.sigma
        Cinv_sqrt = np.linalg.inv(np.linalg.cholesky(self.C + 1e-12 * np.eye(self.n)))
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * Cinv_sqrt @ y
        hsig = np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) < (1.4 + 2 / (self.n + 1)) * self.chin
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * y
        ar = (sel - old) / self.sigma
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (not hsig) * self.cc * (2 - self.cc) * self.C) \
            + self.cmu * (ar.T @ np.diag(self.w) @ ar)
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chin - 1))
        self.sigma = float(min(max(self.sigma, 1e-4), 1.0))
        self.gen += 1
        self.pending = []

    def best_mean(self):
        return self._denorm(self.m).tolist()

    def state(self):
        return {"m": self.m.tolist(), "sigma": self.sigma, "C": self.C.tolist(), "pc": self.pc.tolist(),
                "ps": self.ps.tolist(), "gen": self.gen, "seed": self.seed, "asked": self.asked,
                "pending": self.pending, "lo": self.lo.tolist(), "hi": self.hi.tolist()}

    @classmethod
    def from_state(cls, st):
        return cls([0.0] * len(st["m"]), st["lo"], st["hi"], _state=st)
