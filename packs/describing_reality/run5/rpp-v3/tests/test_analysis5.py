"""The run 5 maths on simulated data with known answers (RUN5_BUILD_PLAN step 5), before any Luna run.

Planning values from run 4 (paper/side/run5_noise/noise.json): item noise sigma = 0.18, design surprise tau = 0.01,
n = 360 items (90 bases x 4 k)."""
import numpy as np

from v3 import analysis5 as A

SIGMA, TAU, BASES = 0.18, 0.01, 90
ITEMS = [f"K{k}x{100000 + 500 * b}" for b in range(BASES) for k in (1, 2, 4, 8)]


def simulate(true_means, rng, sigma=SIGMA):
    """scores[d][i] = true mean + item difficulty + noise, clipped to [0, 1]."""
    diff = rng.normal(0, 0.15, len(ITEMS))
    return {d: {i: float(np.clip(m + diff[j] + rng.normal(0, sigma), 0, 1)) for j, i in enumerate(ITEMS)}
            for d, m in true_means.items()}


def test_stage1_finds_the_best_and_the_tie():
    rng = np.random.default_rng(1)
    truth = {f"F{j}": 0.50 + 0.01 * j for j in range(13)}
    truth["TOP1"], truth["TOP2"], truth["A0P"] = 0.75, 0.748, 0.45
    s = A.stage1(simulate(truth, rng), list(truth), ITEMS, control="A0P")
    assert s["winner"] in ("TOP1", "TOP2") and s["runner_up"] in ("TOP1", "TOP2")
    assert s["tie_at_top"]                                            # 0.002 apart: cannot be told apart
    g = s["generated"]["designs"]
    assert g["TOP2"]["could_be_best"] and not g["F0"]["could_be_best"]
    assert s["generated"]["leads_over_control"]["TOP1"]["reject"]       # +0.30 over the control
    assert set(s["by_k"]) == {"1", "2", "4", "8"}


def _null_matrix(rng, effects=None, sigma=SIGMA):
    """32 designs x ITEMS: item difficulty + item noise (+ planted effects); no design surprise (a true null)."""
    rows = A.design_table()
    diff = rng.normal(0, 0.15, len(ITEMS))
    mu = np.array([0.6 + sum(e / 2 * r[f] for f, e in (effects or {}).items()) for r in rows])
    return mu[:, None] + diff[None, :] + rng.normal(0, sigma, (32, len(ITEMS)))


def test_holm_false_alarm_rate_is_about_five_percent():
    """Primary test (across base systems), no real effects: the chance that any of the 9 pre-registered tests says
    yes stays near 5%."""
    rng = np.random.default_rng(2)
    hits, sims = 0, 400
    for _ in range(sims):
        t = A.cluster_tests(_null_matrix(rng), ITEMS, boot=50)
        hits += any(t[n]["holm"]["reject"] for n in A.CONFIRMATORY)
    assert hits / sims <= 0.075, hits / sims


def test_power_for_the_planned_effect():
    """At the noise measured in the pilot (0.26), an effect of 0.014 on one part is found about 80% of the time by the
    primary test (unadjusted)."""
    rng = np.random.default_rng(3)
    found, sims = 0, 300
    for _ in range(sims):
        t = A.cluster_tests(_null_matrix(rng, {"C": 0.014}, sigma=0.26), ITEMS, boot=50)
        found += t["C"]["p"] < 0.05 and t["C"]["effect"] > 0
    assert 0.65 <= found / sims <= 0.95, found / sims


def test_stage2_recovers_planted_effects_and_picks_the_fitted_best():
    rng = np.random.default_rng(4)
    rows = A.design_table()
    names = [f"D{n + 1}" for n in range(32)]
    eff = {"A": 0.04, "C": 0.06, "E": -0.03}                         # mean(+) - mean(-)
    inter = 0.03                                                      # A:C
    truth = {names[n]: 0.55 + sum(e / 2 * rows[n][f] for f, e in eff.items()) + inter / 2 * rows[n]["A"] * rows[n]["C"]
             + rng.normal(0, TAU) for n in range(32)}
    s = A.stage2(simulate(truth, rng), names, ITEMS, boot=500)
    for f, e in eff.items():
        r = s["effects"][f]["range"]
        assert r[0] <= e <= r[1], (f, e, r)
        assert s["effects"][f]["holm"]["reject"]
    assert s["effects"]["A:C"]["range"][0] <= inter <= s["effects"]["A:C"]["range"][1]
    lv = s["final_levels"]
    assert (lv["A"], lv["C"], lv["E"]) == (1, 1, -1)                  # the combination the effects favour
    assert s["residual_df"] == BASES - 1 and s["secondary_16df"]["residual_df"] == 16


def test_parts_without_an_effect_are_rarely_called_real():
    """With real effects on A, C, E and A:C, the parts without one (B, D) are called real in few datasets (Holm)."""
    rows = A.design_table()
    names = [f"D{n + 1}" for n in range(32)]
    false_yes = 0
    for seed in range(100, 160):
        rng = np.random.default_rng(seed)
        truth = {names[n]: 0.55 + 0.02 * rows[n]["A"] + 0.03 * rows[n]["C"] - 0.015 * rows[n]["E"]
                 + 0.015 * rows[n]["A"] * rows[n]["C"] + rng.normal(0, TAU) for n in range(32)}
        Y, _, _ = A.matrix(simulate(truth, rng), names, ITEMS)
        f = A.fit_factorial(Y.mean(axis=1))
        false_yes += any(f["effects"][n]["holm"]["reject"] for n in ("B", "D"))
    assert false_yes / 60 <= 0.10, false_yes


def test_final_check_passes_only_for_a_real_lead():
    rng = np.random.default_rng(5)
    # V3-058: ids outside the retired sealed set (analysis5 refuses those)
    items720 = [f"K{k}x{2600000 + 500 * b}" for b in range(180) for k in (1, 2, 4, 8)]
    diff = rng.normal(0, 0.15, len(items720))
    sc = {d: {i: float(np.clip(m + diff[j] + rng.normal(0, SIGMA), 0, 1)) for j, i in enumerate(items720)}
          for d, m in {"FIN": 0.70, "WIN": 0.66, "RUN": 0.70, "A0P": 0.5}.items()}
    f = A.final(sc, "FIN", "WIN", "RUN", "A0P", items720)
    assert f["sealed"]["vs_winner"]["passes"]                         # +0.04 on 720 items
    assert not f["sealed"]["vs_runner_up"]["passes"]                  # no real lead


def test_cluster_bootstrap_resamples_whole_bases():
    W = A._weights(ITEMS[:8], boot=50)
    assert np.all(W[:, 0] == W[:, 3]) and np.all(W[:, 4] == W[:, 7])  # the four k of a base move together
