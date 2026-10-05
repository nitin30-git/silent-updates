# Silent Updates

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23157456.svg)](https://doi.org/10.5281/zenodo.23157456)

**A Hugging Face model name is not a version.** `from_pretrained("org/model")` loads whatever is on
`main` today. When the owner pushes new weights to the same repository, every user who has not
pinned a revision gets them on the next download, with no warning.

This repository contains the data, code and paper for a study measuring how often that happens and
what it does to predictions. It also includes a small checker you can put in CI.

## What we found

Over the 962 most downloaded text-classification models on the Hub, comparing each model's release
state with its current state on the same 2,872 inputs:

| What happened in the repository | Models | Changed at least one prediction |
|---|---:|---:|
| Checkpoint rewritten in place | 46 | **43 (93.5%)** |
| Checkpoint rewritten and format switched | 16 | **10 (62.5%)** |
| `safetensors` conversion by the Hub bot | 112 | 0 |
| `config.json` or tokenizer edit only | 42 | 0 |

- **"The files changed" is the wrong alarm.** 280 of 875 models (32%) changed a file that affects
  inference, but most of those changes are bit-exact format conversions. A check on file hashes is a
  false alarm about seven times in ten.
- **Genuine rewrites are common across the Hub.** 11.4% of the main sample, and 12.5% of 1,499
  further models across six other pipeline tags, had their checkpoint rewritten in place.
- **The direction is unpredictable.** Re-evaluated on gold-labelled data from their own task, 17
  changed models produced 6 significant improvements and 2 significant regressions (4 and 2 after
  Holm correction). One financial-sentiment model lost 12 accuracy points.
- **Accuracy can hide the change.** One prompt-injection guard showed no significant accuracy change
  while the share of attacks it missed rose from 25.3% to 51.3%.
- **Almost nobody pins.** None of 1,783 Hub Spaces naming a changed model pins a revision, and 1.9%
  of 7,720 matching GitHub code fragments do.

All numbers are a snapshot of the Hub on 1 October 2026.

## Protect your own code

Pin the revision:

```python
from transformers import pipeline
clf = pipeline("text-classification", model="org/model",
               revision="<40-character commit hash>")
```

Then use the checker to know when an upgrade is worth testing. It needs no GPU and downloads no
weights:

```bash
pip install "git+https://github.com/nitin30-git/silent-updates#subdirectory=tool"

silent-updates check org/model --since <commit-you-pinned>   # what changed since my pin?
silent-updates pin org/model                                 # give me a line to pin today's version
silent-updates audit models.txt                              # check every model in a project
```

It reports `UNCHANGED`, `CONVERTED`, `METADATA` or `REWRITTEN` and exits non-zero only on
`REWRITTEN`, so routine conversions do not break your build. See [`tool/README.md`](tool/README.md)
and the CI example in [`tool/examples/`](tool/examples/).

## Repository layout

```
paper/      LaTeX source and figures
collect/    data collection scripts (run on Kaggle or Colab)
analyse/    analysis scripts that turn the data into every number in the paper
tool/       the silent-updates checker
data/       collected data and computed statistics (see data/README.md)
```

## Reproducing the paper

The analysis runs on a laptop in under a minute and needs only `numpy`, `scipy` and `matplotlib`:

```bash
pip install -r requirements.txt
cd data
python ../analyse/analyze.py          # RQ1-RQ3 core numbers      -> stats.json
python ../analyse/analyze2.py         # flip-rate details          -> stats2.json
python ../analyse/analyze3.py         # input-set robustness
python ../analyse/analyze4.py         # predictors of change       -> stats3.json
python ../analyse/analyze5.py         # safety classifiers         -> stats_safety.json
python ../analyse/analyze6.py         # survival, grace periods, RQ5 -> stats_extra.json
python ../analyse/analyze_gh_tags.py  # GitHub pinning, other tags -> stats_gh_tags.json
python ../analyse/figure.py           # Fig. 1
python ../analyse/figure2.py          # accuracy figure
```

The computed `stats*.json` files are committed, so you can compare your run against them.

Collecting the data again needs a free Hugging Face token (and a GitHub token for the code search),
stored as notebook secrets named `HF_TOKEN` and `GITHUB_TOKEN`, never pasted into code. Each script in
`collect/` documents its platform, runtime and setup at the top. Re-collecting gives today's Hub,
not ours, so expect the numbers to drift; that is the point of the study.

No repository code is ever executed: all models are loaded with `trust_remote_code=False`.

## A note on the models named here

The data names specific models whose predictions changed. An update is not a fault: owners are
entitled to improve their models, and many updates here are improvements. The problem we measure is
that downstream users are not told and are not protected by default.

## Citing

See [`CITATION.cff`](CITATION.cff).

> N. Sarvesh S R, "Silent Updates: In-Place Revisions of Hugging Face Models Change What Deployed
> Classifiers Predict," 2026.

## Licence

Code and data are released under the MIT licence. Model and dataset names belong to their owners.
