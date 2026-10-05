#!/usr/bin/env python3
"""Six additional analyses on the data already collected. No new runs.

1. Survival: how long does a published model stay as released?
2. Predictors: do popularity or age predict a rewrite?
3. A significance test on the paper's central dichotomy.
4. Multiple-comparison correction for the 17 RQ5 tests.
5. Validation of the rewrite-vs-conversion classifier against commit messages.
6. What a user gives up by pinning: are rewrites fixes or new behaviour?
"""
import json, re, math, collections
import numpy as np
from datetime import datetime, timedelta
from scipy import stats

D = "su_v5"
CL = json.load(open(f"{D}/classified.json"))
RES = json.load(open(f"{D}/flip_results_v5.json"))
MODELS = json.load(open(f"{D}/models.json"))
DL = {m["id"]: m["downloads"] for m in MODELS}
LIKES = {m["id"]: m.get("likes", 0) for m in MODELS}
TL = {}
for line in open(f"{D}/timelines.jsonl"):
    x = json.loads(line); TL[x["id"]] = x
ts = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
GRACE, OUT = 7, {}

# ---------------------------------------------------------------- 1. survival
# Event: the first post-release commit that rewrites the stored checkpoint, judged per commit
# (same effective format with different content, or a format switch whose .bin also changed).
# Models with no such commit are right-censored at the collection date.
print("=" * 86); print("1. HOW LONG DOES A MODEL STAY AS RELEASED?"); print("=" * 86)
COLLECT = max(ts(x["head_t"]) for x in TL.values() if x.get("head_t"))
dur, event = [], []
for mid, x in TL.items():
    ev = x.get("events") or []
    if not ev or not x.get("head"): continue
    cut = ts(ev[0]["t"]) + timedelta(days=GRACE)
    rel = [e for e in ev if ts(e["t"]) <= cut][-1]
    t0 = ts(rel["t"]); prev = rel; hit = None
    for e in [e for e in ev if ts(e["t"]) > cut]:
        if e["w"] != prev["w"] and (e["fmt"] == prev["fmt"] or e["wbin"] != prev["wbin"]):
            hit = ts(e["t"]); break
        prev = e
    if hit: dur.append((hit - t0).days); event.append(1)
    else:   dur.append((ts(x["head_t"]) - t0).days if ts(x["head_t"]) > t0 else 0); event.append(0)
dur, event = np.array(dur, float), np.array(event)
order = np.argsort(dur); dur, event = dur[order], event[order]
# Kaplan-Meier
times, S, n = [], [], len(dur)
surv = 1.0; at_risk = n
for t in np.unique(dur):
    d = int(((dur == t) & (event == 1)).sum()); c = int(((dur == t) & (event == 0)).sum())
    if at_risk > 0 and d > 0:
        surv *= (1 - d / at_risk); times.append(float(t)); S.append(surv)
    at_risk -= (d + c)
times, S = np.array(times), np.array(S)
def surv_at(days):
    if len(times) == 0: return 1.0
    i = np.searchsorted(times, days, side="right") - 1
    return float(S[i]) if i >= 0 else 1.0
print(f"models: {n} | rewritten: {int(event.sum())} | still as released: {int((event==0).sum())}")
for d_ in (180, 365, 730, 1095):
    print(f"  still as released after {d_:4d} days: {100*surv_at(d_):.1f}%")
half = times[S <= .5]
print("  median time to rewrite: " + (f"{half[0]:.0f} days" if len(half) else
      "not reached (fewer than half of models are ever rewritten)"))
OUT["survival"] = dict(n=n, events=int(event.sum()),
                       s180=round(100*surv_at(180),1), s365=round(100*surv_at(365),1),
                       s730=round(100*surv_at(730),1), s1095=round(100*surv_at(1095),1),
                       median_days=(float(half[0]) if len(half) else None))
np.savez("survival.npz", times=times, S=S, dur=dur, event=event)

# ---------------------------------------------------------------- 2. predictors
print("\n" + "=" * 86); print("2. DOES POPULARITY OR AGE PREDICT A REWRITE?"); print("=" * 86)
def kind(c):
    if not c["weights_differ"]: return "meta" if (c["config_differs"] or c["tok_differs"]) else "none"
    return "rewrite" if (c["same_format"] or c["bin_differs"]) else "convert"
ids = [m for m in CL]
y = np.array([kind(CL[m]) == "rewrite" for m in ids])
dls = np.array([DL.get(m, 0) for m in ids], float)
age = np.array([(COLLECT - ts(CL[m]["t_first"])).days for m in ids], float)
q = np.quantile(dls, [.25, .5, .75])
print(f"{'download quartile':<22s} {'n':>5s} {'rewritten':>10s} {'rate':>7s}")
bins = [(0, q[0]), (q[0], q[1]), (q[1], q[2]), (q[2], np.inf)]
rates = []
for lo, hi in bins:
    sel = (dls >= lo) & (dls < hi)
    k = int(y[sel].sum()); nn = int(sel.sum()); rates.append((k, nn))
    print(f"{f'{lo:,.0f} - {hi:,.0f}'[:22]:<22s} {nn:5d} {k:10d} {100*k/max(nn,1):6.1f}%")
tab = np.array([[k, nn - k] for k, nn in rates])
chi2, p_dl = stats.chi2_contingency(tab)[:2]
print(f"  chi-square across quartiles: p = {p_dl:.3f}"
      + ("  -> popularity does not predict rewriting" if p_dl > .05 else "  -> popularity predicts rewriting"))
r_age, p_age = stats.pointbiserialr(y.astype(int), age)
print(f"  age vs rewrite: r = {r_age:+.3f}, p = {p_age:.3g}"
      f"  (median age rewritten {np.median(age[y]):.0f}d vs unchanged {np.median(age[~y]):.0f}d)")
OUT["predictors"] = dict(p_downloads=round(float(p_dl), 4), r_age=round(float(r_age), 3),
                         p_age=float(p_age), quartiles=[[int(k), int(nn)] for k, nn in rates])

# ---------------------------------------------------------------- 3. the dichotomy, tested
print("\n" + "=" * 86); print("3. IS THE CENTRAL DICHOTOMY STATISTICALLY SOLID?"); print("=" * 86)
ok = {m: r for m, r in RES.items() if "flip_idx" in r and not r.get("random_head")}
real = [m for m in ok if m in CL and kind(CL[m]) == "rewrite"]
inert = [m for m in ok if m in CL and kind(CL[m]) in ("convert", "meta")]
a = sum(ok[m]["flip_idx"] > 0 for m in real); b = len(real) - a
c = sum(ok[m]["flip_idx"] > 0 for m in inert); d_ = len(inert) - c
odds, p_fish = stats.fisher_exact([[a, b], [c, d_]])
print(f"  genuine rewrite : {a}/{len(real)} changed predictions")
print(f"  conversion/meta : {c}/{len(inert)} changed predictions")
print(f"  Fisher exact two-sided p = {p_fish:.3g}")
OUT["dichotomy"] = dict(rewrite=[a, len(real)], inert=[c, len(inert)], p=float(p_fish))

# ---------------------------------------------------------------- 4. multiple comparisons
print("\n" + "=" * 86); print("4. RQ5 WITH A MULTIPLE-COMPARISON CORRECTION"); print("=" * 86)
try:
    G = json.load(open("gold_results.json"))
    gi = {r["id"] for r in json.load(open("gold_inf.json"))}
    rows = [(m, r["acc_R"], r["acc_L"], r["mcnemar_p"]) for m, r in G.items()
            if "acc_R" in r and m in gi]
    rows.sort(key=lambda r: r[3])
    k = len(rows); surv_h = True
    print(f"{'model':<48s} {'delta':>7s} {'raw p':>10s} {'Holm p':>10s} {'sig':>5s}")
    hp, prev = [], 0.0
    for i, (m, ar, al, p) in enumerate(rows):
        adj = min(1.0, max(prev, (k - i) * p)); prev = adj; hp.append(adj)
        print(f"{m[:48]:<48s} {al-ar:+7.1%} {p:10.3g} {adj:10.3g} {'yes' if adj < .05 else '':>5s}")
    nsig = sum(x < .05 for x in hp)
    up = sum(1 for (m, ar, al, p), x in zip(rows, hp) if x < .05 and al > ar)
    dn = nsig - up
    print(f"\n  after Holm correction: {nsig} of {k} significant ({up} improved, {dn} degraded)")
    OUT["rq5_holm"] = dict(k=k, significant=nsig, improved=up, degraded=dn)
except FileNotFoundError:
    print("  gold results not present; skipped")

# ---------------------------------------------------------------- 5. classifier validation
print("\n" + "=" * 86); print("5. DOES THE REWRITE/CONVERSION LABEL MATCH THE COMMIT MESSAGE?"); print("=" * 86)
CONV = re.compile(r"safetensors|convert|SFconvertbot", re.I)
agree = disagree = 0; examples = []
for m, c in CL.items():
    if not c["weights_differ"] or not c["w_changes"]: continue
    k = kind(c); title = c["w_changes"][0]["title"]; says_conv = bool(CONV.search(title))
    expect_conv = (k == "convert")
    if says_conv == expect_conv: agree += 1
    else:
        disagree += 1
        if len(examples) < 8: examples.append((k, title[:58], m[:38]))
tot = agree + disagree
print(f"  first weight-change commit message agrees with our label: {agree}/{tot} ({100*agree/tot:.1f}%)")
print("  disagreements (these are expected: a repo can be converted AND later rewritten):")
for k, t, m in examples: print(f"    [{k:<8s}] {t:<58s} {m}")
OUT["classifier_check"] = dict(agree=agree, total=tot, pct=round(100*agree/tot, 1))

# ---------------------------------------------------------------- 6. the cost of pinning
print("\n" + "=" * 86); print("6. WHAT DO YOU GIVE UP BY PINNING?"); print("=" * 86)
FIX = re.compile(r"\bfix|\bbug|\bcorrect|\bpatch|\bsecur|\bvuln|\brepair|\bissue\b", re.I)
IMPROVE = re.compile(r"improv|better|updat|retrain|new model|v\d|fine.?tun|epoch", re.I)
cats = collections.Counter(); seen = 0
for c in CL.values():
    if kind(c) != "rewrite": continue
    for w in c["w_changes"]:
        seen += 1
        t = w["title"]
        cats["explicit fix" if FIX.search(t) else
             "retrain / improvement" if IMPROVE.search(t) else "unexplained"] += 1
for k, v in cats.most_common():
    print(f"  {k:<24s} {v:5d}  {100*v/seen:5.1f}%")
print("  (commit messages only; a message is not evidence that the change did what it says)")
OUT["pin_cost"] = {k: int(v) for k, v in cats.items()}

json.dump(OUT, open("stats_extra.json", "w"), indent=1)
print("\nwrote stats_extra.json and survival.npz")
