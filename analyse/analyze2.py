#!/usr/bin/env python3
"""Subgroup analysis: which KIND of change actually changes behaviour, and does accuracy move?"""
import json, numpy as np, collections

D = "su_v5"
CL = json.load(open(f"{D}/classified.json"))
RES = json.load(open(f"{D}/flip_results_v5.json"))
T = json.load(open(f"{D}/texts.json"))
DL = {m["id"]: m["downloads"] for m in json.load(open(f"{D}/models.json"))}
ok = {k: v for k, v in RES.items() if "flip_idx" in v and not v.get("random_head")}
S = {}

print("=" * 86)
print("WHICH KIND OF CHANGE MOVES PREDICTIONS?")
print("=" * 86)
groups = collections.defaultdict(list)
for m, r in ok.items():
    c = CL.get(m)
    if not c: continue
    if c["weights_differ"]:
        g = "weights overwritten (same format)" if c["same_format"] else "format switch only (bin->safetensors)"
    elif c["config_differs"] or c["tok_differs"]:
        g = "config/tokenizer only"
    else:
        g = "no recorded change"
    groups[g].append(r["flip_idx"])

print(f"{'group':<40s} {'n':>4s} {'flip>0':>7s} {'flip>10%':>9s} {'median':>8s} {'max':>7s}")
for g, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    a = np.array(v)
    print(f"{g:<40s} {len(a):4d} {(a>0).sum():7d} {(a>.10).sum():9d} {np.median(a):8.1%} {a.max():7.1%}")
    S[g] = dict(n=len(a), gt0=int((a > 0).sum()), gt10=int((a > .10).sum()),
                median=float(np.median(a)), max=float(a.max()))

fmt_only = [m for m, r in ok.items() if CL.get(m, {}).get("weights_differ")
            and not CL[m]["same_format"] and r["flip_idx"] > 0]
print(f"\nformat-switch-only models that still flip: {len(fmt_only)}")
for m in fmt_only[:10]: print(f"   {m}  {ok[m]['flip_idx']:.1%}")

print("\n" + "=" * 86)
print("SST-2 ACCURACY: DOES THE LATEST VERSION GET BETTER OR WORSE?")
print("=" * 86)
gold = np.array([t["gold"] if t["gold"] is not None else -1 for t in T])
mask = gold >= 0
g = gold[mask]
rows = []
for m, r in ok.items():
    if r["flip_idx_sst"] == 0: continue
    try:
        z = np.load(f"{D}/preds/{m.replace('/', '__')}.npz")
        PR, PL = z["R"][mask].astype(np.float32), z["L"][mask].astype(np.float32)
    except Exception: continue
    if PR.shape[1] != 2: continue                      # need a binary head to map onto SST-2
    for flip in (False, True):                          # label order is not guaranteed; take the better mapping
        aR = (PR.argmax(1) ^ int(flip)); aL = (PL.argmax(1) ^ int(flip))
        accR, accL = (aR == g).mean(), (aL == g).mean()
        if flip == 0: best = (accR, accL)
        elif accR > best[0]: best = (accR, accL)
    if best[0] < 0.70: continue                         # not an English sentiment model; skip
    rows.append((m, best[0], best[1], r["flip_idx_sst"], DL.get(m, 0)))

rows.sort(key=lambda x: x[2] - x[1])
print(f"binary English-sentiment models with SST-2 flips: {len(rows)}")
print(f"{'model':<48s} {'accR':>6s} {'accL':>6s} {'delta':>7s} {'flip':>6s}")
for m, a, b, fl, d in rows:
    print(f"{m[:48]:<48s} {a:6.1%} {b:6.1%} {b-a:+7.1%} {fl:6.1%}")
if rows:
    worse = sum(1 for _, a, b, _, _ in rows if b < a - .005)
    better = sum(1 for _, a, b, _, _ in rows if b > a + .005)
    S["sst2"] = dict(n=len(rows), worse=worse, better=better, same=len(rows) - worse - better,
                     mean_delta=float(np.mean([b - a for _, a, b, _, _ in rows])))
    print(f"\nlatest is WORSE on SST-2: {worse} | BETTER: {better} | unchanged: {len(rows)-worse-better}")
    print(f"mean accuracy change: {S['sst2']['mean_delta']:+.1%}")

print("\n" + "=" * 86)
print("EXPOSURE OF THE MODELS THAT ACTUALLY FLIP")
print("=" * 86)
fl = [(m, r["flip_idx"], DL.get(m, 0)) for m, r in ok.items() if r["flip_idx"] > .01]
S["dl_flippers"] = sum(d for _, _, d in fl)
S["dl_evaluated"] = sum(DL.get(m, 0) for m in ok)
print(f"models flipping >1%: {len(fl)} | their combined monthly downloads: {S['dl_flippers']:,}")
print(f"(evaluated set total downloads: {S['dl_evaluated']:,})")
for m, f_, d in sorted(fl, key=lambda x: -x[2])[:8]:
    print(f"   {m[:56]:<56s} {f_:6.1%} {d:12,} dl/mo")
json.dump(S, open("stats2.json", "w"), indent=1)
print("\nwrote stats2.json")
