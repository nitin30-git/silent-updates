#!/usr/bin/env python3
"""Why do 'format switch only' models flip? If the .bin never changed but the new .safetensors
behaves differently, the conversion did not come from the published checkpoint."""
import json, numpy as np

D = "su_v5"
CL = json.load(open(f"{D}/classified.json"))
RES = json.load(open(f"{D}/flip_results_v5.json"))
DL = {m["id"]: m["downloads"] for m in json.load(open(f"{D}/models.json"))}
ok = {k: v for k, v in RES.items() if "flip_idx" in v and not v.get("random_head")}

print("format-switch models that flip: did the legacy .bin ALSO change?")
print(f"{'model':<52s} {'flip':>6s} {'bin changed':>12s} {'cfg':>5s} {'tok':>5s}  first change title")
n_bin_same = 0
for m, r in sorted(ok.items(), key=lambda kv: -kv[1]["flip_idx"]):
    c = CL.get(m)
    if not c or not c["weights_differ"] or c["same_format"] or r["flip_idx"] == 0: continue
    title = c["w_changes"][0]["title"][:46] if c["w_changes"] else ""
    if not c["bin_differs"]: n_bin_same += 1
    print(f"{m[:52]:<52s} {r['flip_idx']:5.1%} {str(c['bin_differs']):>12s} "
          f"{str(c['config_differs']):>5s} {str(c['tok_differs']):>5s}  {title}")
print(f"\n-> {n_bin_same} of them kept an IDENTICAL .bin: the safetensors file does not match "
      f"the checkpoint it supposedly converts.")

print("\n" + "=" * 86)
print("WHO MAKES THE CHANGES (commit author + message of first post-release weight change)")
print("=" * 86)
import collections, re
auth = collections.Counter(); bot = 0; tot = 0
for c in CL.values():
    for w in c["w_changes"]:
        tot += 1; auth[w["author"]] += 1
        if re.search(r"safetensors|convert|SFconvertbot|bot", w["author"] + " " + w["title"], re.I): bot += 1
print(f"weight-change commits: {tot} | matching a bot/conversion pattern: {bot} ({100*bot/tot:.1f}%)")
for a, k in auth.most_common(8): print(f"   {a:<34s} {k}")

print("\ncommon commit messages:")
msg = collections.Counter(w["title"][:52] for c in CL.values() for w in c["w_changes"])
for t, k in msg.most_common(10): print(f"   {k:4d}  {t}")

print("\n" + "=" * 86)
print("BINARY-HEAD COVERAGE FOR THE SST-2 ACCURACY CHECK")
print("=" * 86)
T = json.load(open(f"{D}/texts.json"))
gold = np.array([t["gold"] if t["gold"] is not None else -1 for t in T]); mask = gold >= 0; g = gold[mask]
n2 = n2_good = 0
for m in ok:
    try: z = np.load(f"{D}/preds/{m.replace('/', '__')}.npz")
    except Exception: continue
    if z["R"].shape[1] != 2: continue
    n2 += 1
    PR = z["R"][mask].astype(np.float32)
    acc = max((PR.argmax(1) == g).mean(), (1 - PR.argmax(1) == g).mean())
    if acc >= .70: n2_good += 1
print(f"evaluated models with a 2-class head: {n2} | of those scoring >=70% on SST-2: {n2_good}")
print("(only these can support an accuracy-direction claim)")
