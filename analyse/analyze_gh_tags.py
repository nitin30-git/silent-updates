"""RQ4 (GitHub dependents and pinning) and the multi-tag prevalence table.
Run from data/:  python ../analyse/analyze_gh_tags.py
Reads gh_results.json and tags_results.json, writes stats_gh_tags.json."""
import json
from scipy.stats import fisher_exact, chi2_contingency


def wilson(k, n, z=1.959964):
    """95% Wilson score interval, in percent."""
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


OUT = {}

# ---------------------------------------------------------------- GitHub
G = json.load(open("gh_results.json"))
groups = {"control": [], "changed": [], "rewritten": []}
for r in G.values():
    groups[r["group"]].append(r)
chg = groups["changed"] + groups["rewritten"]   # all 280 models that changed
ctl = groups["control"]                         # 280 random unchanged models

shown = sum(r["shown"] for r in G.values())
pinned = sum(r["pinned"] for r in G.values())
rw_named = sum(r["files"] > 0 for r in groups["rewritten"])
rw_repos = {repo for r in groups["rewritten"] for repo in r["repos"]}
dep_chg = sum(r["files"] > 0 for r in chg)
dep_ctl = sum(r["files"] > 0 for r in ctl)
def pin_pct(rs):
    return round(100 * sum(r["pinned"] for r in rs) / sum(r["shown"] for r in rs), 1)

OUT["github"] = dict(
    queries=len(G),
    fragments_read=shown, fragments_pinned=pinned,
    pin_pct=round(100 * pinned / shown, 1), pin_ci=wilson(pinned, shown),
    rewritten_models_named=f"{rw_named}/{len(groups['rewritten'])}",
    repos_depending_on_rewritten=len(rw_repos),
    with_dependents_changed=f"{dep_chg}/{len(chg)}",
    with_dependents_control=f"{dep_ctl}/{len(ctl)}",
    dependents_fisher_p=round(fisher_exact([[dep_chg, len(chg) - dep_chg],
                                            [dep_ctl, len(ctl) - dep_ctl]])[1], 3),
    pin_pct_rewritten=pin_pct(groups["rewritten"]),
    pin_pct_converted_or_metadata=pin_pct(groups["changed"]),
    pin_pct_unchanged=pin_pct(ctl),
)

# ---------------------------------------------------------------- other pipeline tags
T = json.load(open("tags_results.json"))
rows, table = [], []
for tag, r in T.items():
    if not r.get("done"):
        continue
    n, k = r["with_weights"], r["counts"]["rewrite"]
    rows.append(dict(tag=tag, n=n, rewritten=k, pct=r["rewrite_pct"], any_change_pct=r["any_pct"]))
    table.append([k, n - k])
N = sum(r["n"] for r in rows)
K = sum(r["rewritten"] for r in rows)
chi2, p_chi, _, _ = chi2_contingency(table)
# the text-classification row against the main study (100 rewritten of 875)
tc = next(r for r in rows if r["tag"] == "text-classification")
p_tc = fisher_exact([[tc["rewritten"], tc["n"] - tc["rewritten"]], [100, 875 - 100]])[1]
OUT["tags"] = dict(rows=sorted(rows, key=lambda r: -r["pct"]), total_n=N, total_rewritten=K,
                   total_pct=round(100 * K / N, 1), total_ci=wilson(K, N),
                   chi2=round(chi2, 1), chi2_p=float(f"{p_chi:.2g}"),
                   text_cls_vs_main_fisher_p=round(p_tc, 3))

json.dump(OUT, open("stats_gh_tags.json", "w"), indent=1)
print(json.dumps(OUT, indent=1))
