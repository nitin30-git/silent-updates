#!/usr/bin/env python3
"""Final analysis for the Silent Updates study. Reads su_v5/, writes stats.json + a printed report."""
import json, math, collections, re
from datetime import datetime, timedelta

D = "su_v5"
MODELS = json.load(open(f"{D}/models.json"))
CL = json.load(open(f"{D}/classified.json"))
RES = json.load(open(f"{D}/flip_results_v5.json"))
SP = json.load(open(f"{D}/spaces_v5.json"))
TL = {}
for line in open(f"{D}/timelines.jsonl"):
    x = json.loads(line)
    TL[x["id"]] = x                      # later line wins (re-fetched after an error)
DL = {m["id"]: m["downloads"] for m in MODELS}
ts = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * p, 100 * max(0, c - h), 100 * min(1, c + h))

def pct(k, n, label):
    p, lo, hi = wilson(k, n)
    return f"{label:<44s} {k:5d}/{n:<5d} {p:5.1f}%  [{lo:4.1f}, {hi:4.1f}]"

S = {}
print("=" * 86)
print("RQ1  PREVALENCE OF IN-PLACE CHANGES")
print("=" * 86)
n_listed = len(MODELS)
n_hist = len([x for x in TL.values() if "error" not in x])
n = len(CL)
S["n_listed"], S["n_hist_ok"], S["n_with_weights"] = n_listed, n_hist, n
print(f"text-classification models listed: {n_listed} | histories fetched: {n_hist} | with loadable weights: {n}")

for key, label in [("weights_differ", "weights differ (release vs latest)"),
                   ("config_differs", "config.json differs"),
                   ("tok_differs", "tokenizer files differ")]:
    k = sum(c[key] for c in CL.values())
    S[key] = k
    print(pct(k, n, label))
k_ov = sum(c["weights_differ"] and c["same_format"] for c in CL.values())
k_fmt = sum(c["weights_differ"] and not c["same_format"] for c in CL.values())
k_any = sum(c["weights_differ"] or c["config_differs"] or c["tok_differs"] for c in CL.values())
S.update(same_format_overwrite=k_ov, format_switch=k_fmt, any_change=k_any)
print(pct(k_ov, n, "  of which same-format overwrite"))
print(pct(k_fmt, n, "  of which format switch (bin->safetensors etc)"))
print(pct(k_any, n, "ANY change to weights/config/tokenizer"))
k_bin = sum(c["bin_differs"] for c in CL.values()); S["bin_differs"] = k_bin
print(pct(k_bin, n, "legacy .bin weights differ"))

# how many post-release weight changes, and how long after release
nch = [c["n_post_weight_changes"] for c in CL.values() if c["weights_differ"]]
S["weight_changes_total"] = sum(nch)
S["weight_changes_median"] = sorted(nch)[len(nch) // 2] if nch else 0
S["weight_changes_max"] = max(nch) if nch else 0
print(f"\npost-release weight-change events: total {sum(nch)}, median per changed model "
      f"{S['weight_changes_median']}, max {S['weight_changes_max']}")
lags = []
for c in CL.values():
    if c["weights_differ"] and c["w_changes"]:
        lags.append((ts(c["w_changes"][0]["t"]) - ts(c["t_R"])).days)
if lags:
    lags.sort()
    S["lag_median_days"], S["lag_p90_days"] = lags[len(lags) // 2], lags[int(.9 * len(lags))]
    S["lag_max_days"] = lags[-1]
    print(f"days from release to first weight change: median {S['lag_median_days']}, "
          f"p90 {S['lag_p90_days']}, max {S['lag_max_days']}")

# README silence: weight changes whose commit did not touch README.md
allw = [w for c in CL.values() for w in c["w_changes"]]
k_silent = sum(not w["readme"] for w in allw)
S["w_changes_all"], S["w_changes_no_readme"] = len(allw), k_silent
print(pct(k_silent, len(allw), "weight changes w/ no README edit in same commit"))

# refs available to pin
k_refs = sum(bool(c["refs"]) for c in CL.values() if c["weights_differ"])
S["changed_with_refs"] = k_refs
print(pct(k_refs, sum(c["weights_differ"] for c in CL.values()), "changed models offering a tag/branch to pin"))

print("\n--- grace-period sensitivity (release = first weights + G days) ---")
def reclassify(x, grace):
    ev = x.get("events") or []
    if not ev or not x.get("head"): return None
    cut = ts(ev[0]["t"]) + timedelta(days=grace)
    rel = [e for e in ev if ts(e["t"]) <= cut][-1]; last = ev[-1]
    return dict(w=rel["w"] != last["w"], c=rel["cfg"] != last["cfg"], t=rel["tok"] != last["tok"])
S["grace"] = {}
for g in (1, 7, 30, 90):
    rr = [r for x in TL.values() if "error" not in x and (r := reclassify(x, g))]
    kw = sum(r["w"] for r in rr); ka = sum(r["w"] or r["c"] or r["t"] for r in rr)
    S["grace"][g] = dict(n=len(rr), weights=kw, any=ka, weights_pct=round(100 * kw / len(rr), 1))
    print(f"  G={g:3d}d  n={len(rr)}  weights differ {kw} ({100*kw/len(rr):.1f}%)  any change {ka} ({100*ka/len(rr):.1f}%)")

print("\n--- download-weighted exposure (monthly downloads of affected repos) ---")
tot_dl = sum(DL.get(m, 0) for m in CL)
ch_dl = sum(DL.get(c["id"], 0) for c in CL.values() if c["weights_differ"])
any_dl = sum(DL.get(c["id"], 0) for c in CL.values() if c["weights_differ"] or c["config_differs"] or c["tok_differs"])
S.update(dl_total=tot_dl, dl_weights_changed=ch_dl, dl_any_changed=any_dl)
print(f"  total monthly downloads in corpus : {tot_dl:,}")
print(f"  on models whose weights changed   : {ch_dl:,} ({100*ch_dl/tot_dl:.1f}%)")
print(f"  on models with any change         : {any_dl:,} ({100*any_dl/tot_dl:.1f}%)")

print("\n" + "=" * 86)
print("RQ2  BEHAVIOURAL IMPACT")
print("=" * 86)
ev = {k: v for k, v in RES.items() if "flip_idx" in v}
rh = {k: v for k, v in RES.items() if v.get("random_head")}
ok = {k: v for k, v in ev.items() if not v.get("random_head")}
fails = {k: v for k, v in RES.items() if v.get("note", "").startswith("infer-fail")}
lsb = {k: v for k, v in RES.items() if v.get("note") == "label space changed"}
S.update(n_attempted=len(RES), n_evaluated=len(ok), n_random_head=len(rh),
         n_fail=len(fails), n_label_space=len(lsb))
print(f"attempted {len(RES)} | evaluated {len(ok)} | random-head excluded {len(rh)} | "
      f"label-space breaks {len(lsb)} | load failures {len(fails)}")

N = len(ok)
f = sorted(ok.values(), key=lambda r: -r["flip_idx"])
for thr, lab in [(0.0, "any index flip (>0%)"), (.01, "flip > 1%"), (.05, "flip > 5%"),
                 (.10, "flip > 10%"), (.25, "flip > 25%"), (.50, "flip > 50%")]:
    k = sum(r["flip_idx"] > thr for r in ok.values())
    S[f"flip_gt_{int(thr*100)}"] = k
    print(pct(k, N, lab))
k_ident = sum(r["flip_idx"] == 0 and r["flip_name"] == 0 for r in ok.values())
k_nameonly = sum(r["flip_idx"] == 0 and r["flip_name"] > 0 for r in ok.values())
S.update(identical=k_ident, name_only=k_nameonly)
print(pct(k_ident, N, "identical predictions AND labels"))
print(pct(k_nameonly, N, "labels renamed only (predictions identical)"))

# hash changed but behaviour identical -> the key "hashes overstate" result
hash_ch = [m for m in ok if m in CL and CL[m]["weights_differ"]]
hash_ch_zero = [m for m in hash_ch if ok[m]["flip_idx"] == 0]
S.update(eval_weights_changed=len(hash_ch), eval_weights_changed_zero_flip=len(hash_ch_zero))
print(pct(len(hash_ch_zero), len(hash_ch), "weights-hash changed BUT zero flips"))
cfgtok = [m for m in ok if m in CL and not CL[m]["weights_differ"]
          and (CL[m]["config_differs"] or CL[m]["tok_differs"])]
cfgtok_fl = [m for m in cfgtok if ok[m]["flip_idx"] > 0]
S.update(eval_cfgtok_only=len(cfgtok), eval_cfgtok_only_flipped=len(cfgtok_fl))
print(pct(len(cfgtok_fl), len(cfgtok), "config/tokenizer-only change AND flips"))

ps = sorted(r["max_prob_shift"] for r in ok.values())
S["prob_shift_median"] = round(ps[len(ps) // 2], 4); S["prob_shift_p90"] = round(ps[int(.9 * len(ps))], 4)
print(f"\nmean max |Delta p| per model: median {S['prob_shift_median']:.3f}, p90 {S['prob_shift_p90']:.3f}")

print("\n--- sensitivity: excluding models that needed a fallback ---")
clean = {k: v for k, v in ok.items() if not v.get("tok_fallback") and not v.get("cfg_patched")}
S["n_clean"] = len(clean)
for thr in (.01, .10):
    k = sum(r["flip_idx"] > thr for r in clean.values())
    S[f"clean_flip_gt_{int(thr*100)}"] = k
    print(pct(k, len(clean), f"clean subset, flip > {int(thr*100)}%"))

print("\n--- top 15 by index flip rate ---")
print(f"{'model':<52s} {'flip':>7s} {'sst':>7s} {'dl/mo':>12s}  labels")
for r in f[:15]:
    lab = "renamed" if r["id2label_R"] != r["id2label_L"] else ""
    print(f"{r['id'][:52]:<52s} {r['flip_idx']:6.1%} {r['flip_idx_sst']:6.1%} {DL.get(r['id'],0):12,}  {lab}")

print("\n--- label-space breaks (output dimension changed) ---")
for k in lsb: print(f"  {k}  ({DL.get(k,0):,} dl/mo)")

print("\n--- failure reasons ---")
cnt = collections.Counter(re.sub(r"infer-fail (\w+).*", r"\1", v["note"]) for v in fails.values())
for k, v in cnt.most_common(): print(f"  {k:<28s} {v}")

print("\n" + "=" * 86)
print("RQ3/RQ4  DETECTABILITY AND DOWNSTREAM EXPOSURE (Spaces)")
print("=" * 86)
allsp = [s for v in SP.values() for s in v if "error" not in s]
used = [(m, s) for m, v in SP.items() for s in v if "error" not in s and s["found_in_code"]]
S.update(n_spaces=len(allsp), n_spaces_models=len(SP), n_spaces_using=len(used))
print(f"models probed {len(SP)} | Spaces inspected {len(allsp)} | model id found in code {len(used)}")
k_pin = sum(s["pinned"] for _, s in used)
S["n_spaces_pinned"] = k_pin
print(pct(k_pin, len(used), "Spaces pinning a revision"))

built_before = stale = 0
for m, s in used:
    c = CL.get(m)
    if not c or not c["w_changes"]: continue
    try:
        ch = ts(c["w_changes"][0]["t"]); cr = ts(s["created"]) if s["created"] else None
        lm = ts(s["last_modified"]) if s["last_modified"] else None
    except Exception: continue
    if cr and cr < ch:
        built_before += 1
        if lm and lm < ch: stale += 1
S.update(spaces_built_before_change=built_before, spaces_never_touched_since=stale)
print(pct(built_before, len(used), "Spaces created BEFORE the model's weight change"))
print(pct(stale, len(used), "  and not modified since that change"))
stage = collections.Counter(s["stage"] for _, s in used)
S["space_stages"] = dict(stage)
print("  runtime stage:", dict(stage.most_common(6)))

json.dump(S, open("stats.json", "w"), indent=1, default=str)
print("\nwrote stats.json")
