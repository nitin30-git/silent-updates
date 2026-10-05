# silent-updates

Tell whether a Hugging Face model's checkpoint was **rewritten in place** since you pinned it.

`from_pretrained("org/model")` loads whatever is on `main` today. The owner can push new weights to
the same repository, and you get them on the next download with no warning. This tool answers the
only question that matters for reproducibility, without a GPU and without downloading any weights:

> Has the stored checkpoint actually been rewritten, or was the change just a format conversion?

That distinction matters, and it is why "the files changed" is the wrong alarm to put in CI. In the
study behind this tool, measured over the 962 most downloaded text classifiers:

| What happened in the repository | Models | Changed a prediction |
|---|---|---|
| Checkpoint rewritten in place | 62 | **53 (85.5%)**, median 17% of inputs relabelled |
| `safetensors` conversion by the Hub bot | 112 | **0** |
| `config.json` or tokenizer edit only | 42 | **0** |

A check on changed file hashes would be a false alarm about seven times in ten. This tool
distinguishes the cases by tracking the original `.bin` file separately from the file the library
actually loads: a conversion adds a second copy and leaves the original byte-identical, while a
rewrite does not.

## Install

```bash
pip install "git+https://github.com/YOUR-USERNAME/silent-updates#subdirectory=tool"
```

Needs Python 3.9+ and `git`. Nothing else, and no GPU.

## Use

```bash
# Did anything change since the commit I pinned?
silent-updates check cardiffnlp/twitter-roberta-base-sentiment --since <commit-you-pinned>

# Give me a line I can paste to pin the current version
silent-updates pin cardiffnlp/twitter-roberta-base-sentiment

# Check every model my project depends on
silent-updates audit models.txt
```

`models.txt` holds one `model [revision]` per line; `#` starts a comment. With no revision, the tool
compares against the model's *release state*: the last commit within 7 days of the first commit that
contains weights (most repositories start with a commit that holds no weights at all).

```
cardiffnlp/twitter-roberta-base-sentiment  <commit>
unitary/toxic-bert                         <commit>
# protectai/deberta-v3-base-prompt-injection-v2   not pinned yet
```

## Verdicts and exit codes

| Verdict | Exit | Meaning |
|---|---|---|
| `UNCHANGED` | 0 | Weights, config and tokenizer are all identical. |
| `CONVERTED` | 0 | A `safetensors` copy was added; the original checkpoint is byte-identical. No model in the study changed a prediction this way. |
| `METADATA` | 0 | Weights identical, but config or tokenizer changed. No prediction changed this way in the study, but **label names may have changed**, which breaks code comparing label strings. |
| `REWRITTEN` | 1 | The stored checkpoint was rewritten. Re-test before adopting. |
| `UNKNOWN` | 2 | No loadable weight files at one of the two revisions. |

Only `REWRITTEN` fails, so wiring this into CI does not break your build every time the conversion
bot runs.

## In CI

```yaml
# .github/workflows/model-drift.yml
name: model drift
on:
  schedule: [{cron: "0 6 * * 1"}]   # Mondays
  workflow_dispatch:
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install "git+https://github.com/YOUR-USERNAME/silent-updates#subdirectory=tool"
      - run: silent-updates audit models.txt
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}   # only needed for gated or private repos
```

The job fails when a dependency's checkpoint has been rewritten, and tells you the commit to pin.

## What it does not do

It does not tell you whether the new version is **better**. In the study, re-evaluating 17 changed
models on gold-labelled data from their own task found 6 significant improvements and 2 significant
regressions (4 and 2 after Holm correction), so the direction is genuinely unpredictable and you have to measure it on your own data.
It also found that aggregate accuracy can hide the change that matters: one prompt-injection guard
showed no significant accuracy change while the share of attacks it let through went from 25.3% to
51.3%. If your classifier gates something, test its error profile, not just its accuracy.

## How it works

1. `git clone --bare --filter=blob:none` fetches commits and file trees but no weights.
2. For the two revisions it reads the root tree and records, for each, the blob id of every file
   `transformers` loads: `model*.safetensors` / `pytorch_model*.bin` (including shards),
   `config.json` and the tokenizer files. Git LFS pointer blobs change exactly when the stored file
   changes, so this is exact without fetching anything large.
3. It compares the *effective* weight set (safetensors when present, else `.bin`, as the library
   chooses) and separately the `.bin` history, which is what separates a conversion from a rewrite.

`training_args.bin`, optimizer state and similar files are ignored: they change on every training
run and never reach a user.

## Citing

If you use this in work you publish, please cite the paper (see `CITATION.cff`).

## Licence

MIT.
