#!/usr/bin/env python3
"""Correct taxonomy: separate a pure format conversion (behaviour-preserving by construction)
from a genuine rewrite of the checkpoint. 'bin_differs' tells them apart: a conversion bot adds
a .safetensors file and leaves the original .bin untouched."""
import json, numpy as np, math

D = "su_v5"
CL = json.load(open(f"{D}/classified.json"))
RES = json.load(open(f"{D}/flip_results_v5.json"))
DL = {m["id"]: m["downloads"] for m in json.load(open(f"{D}/models.json"))}
ok = {k: v for k, v in RES.items() if "flip_idx" in v and not v.get("random_head")}

def wilson(k, n, z=1.96):
    if n == 0: return (0., 0., 0.)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * p, 100 * max(0, c - h), 100 * min(1, c + h))

def kind(c):
    if not c["weights_differ"]:
        return "C. config/tokenizer only" if (c["config_differs"] or c["tok_differs"]) else "D. no change"
    if c["same_format"]: return "A. rewritten in place (same format)"
    return "B. rewritten + format switch" if c["bin_differs"] else "B0. pure format conversion (bot)"

print("=" * 92)
print("CORPUS-LEVEL: how the 875 models with weights break down")
print("=" * 92)
import collections
pop = collections.Counter(kind(c) for c in CL.values())
dlp = collections.Counter()
for c in CL.values(): dlp[kind(c)] += DL.get(c["id"], 0)
for g in sorted(pop):
    p, lo, hi = wilson(pop[g], len(CL))
    print(f"{g:<42s} {pop[g]:4d}  {p:5.1f}%  [{lo:4.1f},{hi:4.1f}]   {dlp[g]:12,} dl/mo")

print("\n" + "=" * 92)
print("BEHAVIOUR BY KIND OF CHANGE (evaluated models, random heads excluded)")
print("=" * 92)
G = collections.defaultdict(list)
for m, r in ok.items():
    if m in CL: G[kind(CL[m])].append((m, r["flip_idx"]))
print(f"{'group':<42s} {'n':>4s} {'flip>0':>14s} {'flip>10%':>14s} {'median':>8s}")
OUT = {}
for g in sorted(G):
    v = np.array([f for _, f in G[g]])
    k0, k10 = int((v > 0).sum()), int((v > .10).sum())
    p0, l0, h0 = wilson(k0, len(v)); p10, l10, h10 = wilson(k10, len(v))
    print(f"{g:<42s} {len(v):4d} {k0:4d} {p0:5.1f}% {'':1s} {k10:4d} {p10:5.1f}% {'':3s} {np.median(v):7.1%}")
    OUT[g] = dict(n=len(v), gt0=k0, gt0_pct=round(p0, 1), gt0_ci=[round(l0, 1), round(h0, 1)],
                  gt10=k10, gt10_pct=round(p10, 1), gt10_ci=[round(l10, 1), round(h10, 1)],
                  median=round(float(np.median(v)), 4))

A = "A. rewritten in place (same format)"; B = "B. rewritten + format switch"
B0 = "B0. pure format conversion (bot)"; C = "C. config/tokenizer only"
real = [f for g in (A, B) for _, f in G.get(g, [])]
inert = [f for g in (B0, C) for _, f in G.get(g, [])]
kr, nr = sum(f > 0 for f in real), len(real)
ki, ni = sum(f > 0 for f in inert), len(inert)
pr, lr, hr = wilson(kr, nr); pi, li, hi2 = wilson(ki, ni)
print(f"\nGENUINE REWRITE   : {kr}/{nr} change predictions  {pr:.1f}% [{lr:.1f}, {hr:.1f}]")
print(f"CONVERSION/METADATA: {ki}/{ni} change predictions  {pi:.1f}% [{li:.1f}, {hi2:.1f}]")
OUT["genuine_rewrite"] = dict(k=kr, n=nr, pct=round(pr, 1), ci=[round(lr, 1), round(hr, 1)])
OUT["inert"] = dict(k=ki, n=ni, pct=round(pi, 1), ci=[round(li, 1), round(hi2, 1)])

# corpus-level projection: share of all models whose checkpoint was genuinely rewritten
n_real = pop[A] + pop[B]
p, lo, hi = wilson(n_real, len(CL))
OUT["corpus_genuine_rewrite"] = dict(k=n_real, n=len(CL), pct=round(p, 1), ci=[round(lo, 1), round(hi, 1)])
OUT["dl_genuine_rewrite"] = dlp[A] + dlp[B]
print(f"\ncorpus: {n_real}/{len(CL)} ({p:.1f}% [{lo:.1f}, {hi:.1f}]) had the checkpoint genuinely rewritten")
print(f"        those repos serve {dlp[A]+dlp[B]:,} downloads/month")

print("\n" + "=" * 92)
print("TRAINING-CHECKPOINT CHURN (repos that push live training state to main)")
print("=" * 92)
churn = [c for c in CL.values() if c["n_post_weight_changes"] >= 10]
OUT["churn_repos"] = len(churn)
print(f"repos with >=10 post-release weight changes: {len(churn)}")
for c in sorted(churn, key=lambda c: -c["n_post_weight_changes"])[:8]:
    print(f"   {c['id'][:56]:<56s} {c['n_post_weight_changes']:4d} changes  {DL.get(c['id'],0):10,} dl/mo")

json.dump(OUT, open("stats3.json", "w"), indent=1)
print("\nwrote stats3.json")
