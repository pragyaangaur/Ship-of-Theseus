"""Recomputes the numbers reported in README.md and RESULTS.md from the raw Kaggle output in results/main."""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "results/main"
NP = 31
ROWS = [json.loads(line) for line in (MAIN / "hybrids.jsonl").open()]
BY = {r["id"]: r for r in ROWS}
A, B = BY["ctrl/A"], BY["ctrl/B"]
GATES = json.loads((MAIN / "gates.json").read_text())


def scores(r, t):
    return np.array([x["score"] for x in r["id_items"] if x["tmpl"] == t])


def iota(r, t="A"):
    a, b = scores(A, t).mean(), scores(B, t).mean()
    return float((a - scores(r, t).mean()) / (a - b))


def phi(r):
    return (A["F_raw"] - r["F_raw"]) / (A["F_raw"] - B["F_raw"])


ARM1 = [r for r in ROWS if r["arm"] == 1]
WORSE = max(A["wiki_ppl"], B["wiki_ppl"])


def test_counts():
    assert len(ROWS) == 261
    assert Counter(r["arm"] for r in ROWS) == {0: 4, 1: 240, 2: 8, 3: 9}
    assert all(r["nonfinite"] == 0 for r in ROWS)


def test_gates():
    assert GATES["g1_pass"] and GATES["g2_pass"] and GATES["g3_pass"] and GATES["g4_pass"]
    assert GATES["g1_A"]["decisive_agree"] == 1.0 and GATES["g1_A"]["n_decisive"] == 32
    assert GATES["g1_B"]["decisive_agree"] == 1.0 and GATES["g1_B"]["n_decisive"] == 26
    assert round(GATES["g1_B"]["argmax_agree"] * 51) == 50
    assert round(GATES["g1_A"]["max_abs_logit_diff"], 2) == 0.79
    assert round(GATES["g1_B"]["max_abs_logit_diff"], 2) == 0.42
    assert round(GATES["g2_sep_A_tmpl"], 1) == 33.1
    assert round(GATES["g2_sep_B_tmpl"], 1) == 26.6
    assert round(GATES["g3_sep"], 2) == 2.35


def test_parents_and_rotary_controls():
    assert round(A["wiki_ppl"], 1) == 12.4 and round(B["wiki_ppl"], 1) == 43.6
    assert round(A["arc_acc"], 2) == 0.95 and round(B["arc_acc"], 2) == 0.83
    assert round(BY["ctrl/A_thetaB"]["wiki_ppl"], 1) == 13.6
    assert round(BY["ctrl/B_thetaA"]["wiki_ppl"], 1) == 63.8


def logistic(k, k50, w):
    return 1 / (1 + np.exp(-(k - k50) / w))


def fit(order, t):
    ks = np.arange(0, NP + 1)
    ys = np.array([0.0] + [iota(BY[f"arm1/{order}/k{k:02d}"], t) for k in range(1, NP)] + [1.0])
    p, _ = curve_fit(logistic, ks, ys, p0=[NP / 2, 2.0], bounds=([-10, 0.05], [NP + 10, 50]), maxfev=5000)
    rss_log = float(((ys - logistic(ks, *p)) ** 2).sum())
    X = np.vstack([ks, np.ones_like(ks)]).T
    coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
    rss_lin = float(((ys - X @ coef) ** 2).sum())
    return p[0], len(ks) * math.log(rss_lin / rss_log)


def test_h1_h2_curves():
    expected = {("A", "forward"): (22.2, -1.2), ("A", "backward"): (5.1, 11.6),
                ("B", "forward"): (26.5, 56.1), ("B", "backward"): (7.4, 1.4)}
    for (t, o), (k50, dbic) in expected.items():
        got_k50, got_dbic = fit(o, t)
        assert abs(got_k50 - k50) < 0.06, (t, o, got_k50)
        assert abs(got_dbic - dbic) < 0.06, (t, o, got_dbic)
    s = json.loads((MAIN / "summary.json").read_text())["H1_H2_curves"]
    assert [round(x, 1) for x in s["A_k50_fwd_minus_bwd_ci"]] == [14.6, 18.6]
    assert [round(x, 1) for x in s["B_k50_fwd_minus_bwd_ci"]] == [17.1, 21.2]


def test_h3_twins_and_additivity():
    masks = {}
    for r in ARM1:
        masks[r["mask"]] = r
    assert len(masks) == 238
    d = []
    for m, r in masks.items():
        comp = "".join("A" if c == "B" else "B" for c in m)
        if comp in masks and m < comp:
            d.append(iota(r) + iota(masks[comp]) - 1)
    assert len(d) == 119
    assert round(float(np.mean(d)), 3) == 0.152
    s = json.loads((MAIN / "summary.json").read_text())
    assert [round(x, 3) for x in s["H3_twins"]["ci"]] == [0.125, 0.177]
    assert round(s["H2_influence"]["iota_A"]["R2"], 2) == 0.85
    assert round(s["H2_influence"]["log_ppl"]["R2"], 2) == 0.17


def test_h4_no_dissociation():
    sea = [r for r in ARM1 if r["wiki_ppl"] <= 2 * WORSE]
    assert len(sea) == 64
    assert sum(abs(iota(r) - phi(r)) > 0.5 for r in sea) == 0


def test_h5_sinking_and_blends():
    fwd = max(BY[f"arm1/forward/k{k:02d}"]["wiki_ppl"] for k in range(1, NP))
    bwd = max(BY[f"arm1/backward/k{k:02d}"]["wiki_ppl"] for k in range(1, NP))
    assert round(fwd) == 29904 and round(bwd) == 86002
    assert round(2 * WORSE) == 87
    blends = [BY[f"arm3/a{j / 10}"]["wiki_ppl"] for j in range(1, 10)]
    assert round(max(blends), 1) == 48.2
    orders = sorted({r["order"] for r in ARM1})
    wins = 0
    for j in range(1, 10):
        k = round(NP * j / 10)
        plank = np.mean([BY[f"arm1/{o}/k{k:02d}"]["wiki_ppl"] for o in orders])
        wins += BY[f"arm3/a{j / 10}"]["wiki_ppl"] < plank
    assert wins == 9


def test_free_self_reports():
    a_tmpl = Counter(g["label"] for r in ARM1 for g in r["gen"] if g["tmpl"] == "A")
    assert a_tmpl == {"neither": 1745, "B": 139, "A": 36}
    assert sum(g["degenerate"] for r in ARM1 for g in r["gen"]) == 2297
    assert sum(1 for r in ARM1 for g in r["gen"] if g["tmpl"] == "A" and g["think_tokens"] > 0) == 275
    assert sum(g["label"] == "A" for g in A["gen"] if g["tmpl"] == "A") + \
        sum(g["label"] == "A" for g in A["gen"] if g["tmpl"] == "B") == 11
    assert any("{{ company }}" in g["text"] for g in BY["arm1/forward/k26"]["gen"])
    assert any("Assistant AI" in g["text"] for g in BY["arm3/a0.5"]["gen"])
    assert round(iota(BY["arm3/a0.1"]), 2) == 0.09
    assert sum(g["label"] in ("A", "both") for g in BY["arm3/a0.1"]["gen"]) == 1


def test_random_order_peak():
    ppl = [BY[f"arm1/random1/k{k:02d}"]["wiki_ppl"] for k in range(1, NP)]
    assert all(p > 1e5 for p in ppl[:12]) and ppl[12] < 1e5
    assert round(max(ppl) / 1e8, 1) == 3.4
