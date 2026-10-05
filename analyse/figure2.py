#!/usr/bin/env python3
"""Figure 2: task accuracy at the release commit and at today's commit, one row per model.
Direction is encoded by the arrow and by an open/filled marker, never by colour alone."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = json.load(open("gold_inf.json"))
rows.sort(key=lambda r: r["accL"] - r["accR"])
SHORT = {"StephanAkkerman/FinTwitBERT-sentiment": "FinTwitBERT-sentiment",
         "pysentimiento/robertuito-sentiment-analysis": "robertuito-sentiment",
         "NeuralTrust/prompt-guard-oss-small": "prompt-guard-oss-small",
         "textdetox/xlmr-large-toxicity-classifier": "xlmr-large-toxicity *",
         "patronus-studio/wolf-defender-prompt-injection-small": "wolf-defender-small",
         "dcarpintero/pangolin-guard-base": "pangolin-guard-base",
         "cardiffnlp/twitter-roberta-base-hate-latest": "twitter-roberta-hate",
         "gbv/mdeberta-ru-prompt-injection": "mdeberta-ru-injection",
         "beethogedeon/Modern-FinBERT-large": "Modern-FinBERT-large",
         "patronus-studio/wolf-defender-prompt-injection": "wolf-defender",
         "finiteautomata/bertweet-base-sentiment-analysis": "bertweet-sentiment",
         "tabularisai/multilingual-sentiment-analysis": "multilingual-sentiment",
         "eliasalbouzidi/distilbert-nsfw-text-classifier": "distilbert-nsfw",
         "finiteautomata/beto-sentiment-analysis": "beto-sentiment",
         "tabularisai/robust-sentiment-analysis": "robust-sentiment",
         "neoyipeng/ModernFinBERT-base": "ModernFinBERT-base",
         "UMUTeam/roberta-spanish-sentiment-analysis": "roberta-spanish-sentiment"}

fig, ax = plt.subplots(figsize=(3.4, 2.95))
INK, DARK = "#888888", "#1a1a1a"
for i, r in enumerate(rows):
    a, b, sig = r["accR"] * 100, r["accL"] * 100, r["p"] < .05
    ax.plot([a, b], [i, i], color=DARK if sig else INK, lw=1.6 if sig else .9,
            solid_capstyle="round", zorder=2)
    ax.scatter([a], [i], s=14, facecolors="white", edgecolors=DARK, linewidths=.9, zorder=3)
    ax.scatter([b], [i], s=14, facecolors=DARK if sig else INK, edgecolors=DARK if sig else INK,
               linewidths=.9, zorder=3)
    if sig:
        ax.text(max(a, b) + 2.5, i, f"{(b-a):+.0f}", va="center", ha="left", fontsize=6, color=DARK)

ax.set_yticks(range(len(rows)))
ax.set_yticklabels([SHORT.get(r["id"], r["id"].split("/")[-1])[:26] for r in rows], fontsize=5.9)
ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", labelsize=7)
ax.set_xlabel("Accuracy on the model's own task (%)", fontsize=7.5)
ax.set_xlim(28, 108); ax.set_ylim(-.8, len(rows) - .2)
ax.grid(axis="x", color="#dddddd", lw=.6, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#999999")
ax.scatter([], [], s=14, facecolors="white", edgecolors=DARK, linewidths=.9, label="release")
ax.scatter([], [], s=14, facecolors=DARK, edgecolors=DARK, linewidths=.9, label="today")
ax.legend(fontsize=6.3, frameon=False, loc="lower left", handletextpad=.3, borderpad=.1)
plt.tight_layout()
plt.savefig("fig_accuracy.pdf", bbox_inches="tight")
plt.savefig("fig_accuracy.png", dpi=200, bbox_inches="tight")
print("rows:", len(rows), "| significant:", sum(r["p"] < .05 for r in rows))
