#!/usr/bin/env python3
"""Figure 1: flip rate by kind of repository change.
Group is encoded by vertical position only, so the figure survives greyscale printing and
needs no legend. One ink color; jitter is deterministic."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "su_v5"
CL = json.load(open(f"{D}/classified.json"))
RES = json.load(open(f"{D}/flip_results_v5.json"))
ok = {k: v for k, v in RES.items() if "flip_idx" in v and not v.get("random_head")}

def kind(c):
    if not c["weights_differ"]:
        return 3 if (c["config_differs"] or c["tok_differs"]) else None
    if c["same_format"]: return 0
    return 1 if c["bin_differs"] else 2

LAB = ["Rewritten in place\n(same format)", "Rewritten\n+ format switch",
       "Format conversion\nonly (bot)", "Config / tokenizer\nonly"]
G = {i: [] for i in range(4)}
for m, r in ok.items():
    c = CL.get(m)
    if c and (k := kind(c)) is not None: G[k].append(r["flip_idx"] * 100)

rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(3.4, 2.5))
INK, ACC = "#444444", "#1a1a1a"
for i in range(4):
    v = np.array(G[i])
    y = 3 - i + rng.uniform(-.17, .17, len(v))
    ax.scatter(v, y, s=9, facecolors="none", edgecolors=INK, linewidths=.6, alpha=.75, zorder=3)
    med = np.median(v) if len(v) else 0
    ax.plot([med, med], [3 - i - .3, 3 - i + .3], color=ACC, lw=1.8, zorder=4)
    ax.text(99, 3 - i + .30, f"{int((v>0).sum())}/{len(v)}", va="bottom", ha="right",
            fontsize=6.5, color=ACC)

ax.set_yticks(range(4)); ax.set_yticklabels(LAB[::-1], fontsize=7)
ax.set_xlabel("Inputs receiving a different predicted class (%)", fontsize=7.5)
ax.set_xlim(-4, 100); ax.set_ylim(-.6, 3.6)
ax.tick_params(axis="x", labelsize=7); ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color="#dddddd", lw=.6, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#999999")
plt.tight_layout()
plt.savefig("fig_flip_by_kind.pdf", bbox_inches="tight")
plt.savefig("fig_flip_by_kind.png", dpi=200, bbox_inches="tight")
print("groups:", {LAB[i].replace(chr(10), ' '): len(G[i]) for i in range(4)})
print("wrote fig_flip_by_kind.pdf / .png")
