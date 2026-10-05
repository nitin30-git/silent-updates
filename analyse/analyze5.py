#!/usr/bin/env python3
"""Safety-classifier subgroup: guards, filters and detectors are the models whose silent change
matters most, because downstream code treats their verdict as a gate."""
import json, re, math, collections

D = "su_v5"
CL = json.load(open(f"{D}/classified.json"))
RES = json.load(open(f"{D}/flip_results_v5.json"))
DL = {m["id"]: m["downloads"] for m in json.load(open(f"{D}/models.json"))}

SAFETY = re.compile(r"inject|jailbreak|guard|toxic|nsfw|hate|safe|moderat|abuse|offens|spam|phish|"
                    r"harm|violen|sexual|profan|threat|detox|shield|filter", re.I)

def wilson(k, n, z=1.96):
    if n == 0: return (0., 0., 0.)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * p, 100 * max(0, c - h), 100 * min(1, c + h))

def kind(c):
    if not c["weights_differ"]:
        return "cfgtok" if (c["config_differs"] or c["tok_differs"]) else "none"
    if c["same_format"]: return "rewrite"
    return "rewrite" if c["bin_differs"] else "convert"

saf = {m: c for m, c in CL.items() if SAFETY.search(m)}
oth = {m: c for m, c in CL.items() if not SAFETY.search(m)}
S = {}
print("=" * 88)
print("SAFETY CLASSIFIERS vs THE REST")
print("=" * 88)
for lab, grp in (("safety classifiers", saf), ("all other models", oth)):
    n = len(grp)
    kr = sum(kind(c) == "rewrite" for c in grp.values())
    p, lo, hi = wilson(kr, n)
    dl = sum(DL.get(m, 0) for m in grp)
    dlr = sum(DL.get(m, 0) for m, c in grp.items() if kind(c) == "rewrite")
    print(f"{lab:<22s} n={n:4d}  checkpoint rewritten {kr:3d} ({p:4.1f}% [{lo:4.1f},{hi:4.1f}])  "
          f"downloads {dl:12,} of which affected {dlr:11,}")
    S[lab] = dict(n=n, rewritten=kr, pct=round(p, 1), ci=[round(lo, 1), round(hi, 1)],
                  downloads=dl, downloads_affected=dlr)

ok = {m: r for m, r in RES.items() if "flip_idx" in r and not r.get("random_head")}
sev = {m: r for m, r in ok.items() if m in saf}
print(f"\nevaluated safety classifiers: {len(sev)}")
k = sum(r["flip_idx"] > 0 for r in sev.values())
p, lo, hi = wilson(k, len(sev))
print(f"  changed at least one verdict: {k} ({p:.1f}% [{lo:.1f},{hi:.1f}])")
S["evaluated_safety"] = dict(n=len(sev), flipped=k, pct=round(p, 1))
print(f"\n{'model':<56s} {'flip':>7s} {'kind':>9s} {'dl/mo':>11s}")
for m, r in sorted(sev.items(), key=lambda kv: -kv[1]["flip_idx"]):
    if r["flip_idx"] == 0: continue
    print(f"{m[:56]:<56s} {r['flip_idx']:6.1%} {kind(CL[m]):>9s} {DL.get(m,0):11,}")
S["flipping_safety"] = [dict(id=m, flip=round(r["flip_idx"], 4), dl=DL.get(m, 0))
                        for m, r in sorted(sev.items(), key=lambda kv: -kv[1]["flip_idx"])
                        if r["flip_idx"] > 0]
S["dl_flipping_safety"] = sum(x["dl"] for x in S["flipping_safety"])
print(f"\ncombined monthly downloads of safety models that changed verdicts: {S['dl_flipping_safety']:,}")

print("\n" + "=" * 88)
print("TASK FAMILIES AMONG THE MODELS THAT CHANGED BEHAVIOUR")
print("=" * 88)
FAM = [("prompt injection / jailbreak", r"inject|jailbreak|prompt.?guard"),
       ("toxicity / hate / NSFW", r"toxic|hate|nsfw|offens|abuse|detox"),
       ("phishing / spam / fraud", r"phish|spam|fraud|scam"),
       ("AI-text / fake-news detection", r"ai.?detect|ai.?gener|fake.?news|detector"),
       ("sentiment / emotion", r"sentiment|emotion|feel|polarity"),
       ("other", r".")]
seen = set()
for lab, pat in FAM:
    rx = re.compile(pat, re.I)
    g = [m for m, r in ok.items() if r["flip_idx"] > .01 and m not in seen and rx.search(m)]
    seen |= set(g)
    if g:
        print(f"{lab:<32s} {len(g):3d} models  {sum(DL.get(m,0) for m in g):11,} dl/mo")
        S.setdefault("families", {})[lab] = dict(n=len(g), dl=sum(DL.get(m, 0) for m in g))

json.dump(S, open("stats_safety.json", "w"), indent=1)
print("\nwrote stats_safety.json")
