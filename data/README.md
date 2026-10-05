# Data

Collected 1 October 2026 (main study and Spaces), 2–4 October 2026 (gold re-evaluation, GitHub,
other pipeline tags). Run the analysis scripts from this folder.

## `su_v5/`: main study (962 text-classification models)

| File | Contents |
|---|---|
| `models.json` | the 962 models: id, monthly downloads, creation date |
| `timelines.jsonl` | one line per model: every commit that changed weights, config or tokenizer, with blob ids |
| `classified.json` | release state vs latest state per model, and the kind of change |
| `flip_results_v5.json` | behavioural comparison for the 261 attempted models (216 evaluated): flip rates, label-space changes, probability shifts, or the reason a model was excluded |
| `preds/*.npz` | raw per-input class probabilities for both versions of each evaluated model |
| `texts.json` | the 2,872 input sentences |
| `spaces_v5.json` | Hub Spaces declaring each changed model, and whether they pin |
| `hand_validation.csv` | 60 Spaces sampled for manually checking the automatic pinning test; the `your_*` columns are left for a human coder |

## Results from the follow-up experiments

| File | Contents |
|---|---|
| `gold_results.json`, `gold_inf.json` | gold-labelled re-evaluation of 21 models (RQ5) |
| `gh_results.json` | GitHub code search for 560 models: matching files, repositories, pinned uses |
| `tags_results.json` | rewrite counts for the top 250 models of each of seven pipeline tags |
| `model_sets.json`, `exp2_models.json` | model lists used to drive the follow-up runs |

## Computed statistics

`stats.json`, `stats2.json`, `stats3.json`, `stats_safety.json`, `stats_extra.json`,
`stats_gh_tags.json` and `survival.npz` are written by the scripts in `../analyse/`. They are
committed so that a re-run can be checked against them.

The data contain only public metadata, model outputs and the names of public repositories. No model
weights are redistributed.
