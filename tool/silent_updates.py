#!/usr/bin/env python3
"""
silent-updates: tell whether a Hugging Face model's checkpoint was rewritten since you pinned it.

Loading a model by name takes whatever is on `main` today. This tool answers, without a GPU and
without downloading any weights, the only question that matters for reproducibility:

    has the stored checkpoint been rewritten, or was the change just a format conversion?

Measurements behind the three verdicts are in the accompanying paper. In the study, a genuine
rewrite changed predictions for 53 of 62 models, while a pure safetensors conversion or a config
or tokenizer edit changed predictions for 0 of 154. So REWRITTEN is worth acting on and CONVERTED
is not, which is why a plain "the files changed" check is the wrong alarm to wire into CI.

    silent-updates check org/model --since <commit>
    silent-updates pin org/model
    silent-updates audit models.txt

Exit codes: 0 nothing to do, 1 checkpoint rewritten, 2 could not determine.
Needs only git and Python 3.9+. No GPU, no weights downloaded (blob-less clone).
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, shutil, subprocess, sys, tempfile

__version__ = "1.0.0"
HUB = "https://huggingface.co"

# Files transformers actually loads. training_args.bin, optimizer state and the like are ignored:
# they change on every training run and never reach a user.
WEIGHT_RE = re.compile(r"^(model|pytorch_model)(-\d{5}-of-\d{5})?\.(safetensors|bin)$")
TOK_RE = re.compile(r"^(tokenizer\.json|tokenizer_config\.json|vocab\.(txt|json)|merges\.txt|"
                    r"spiece\.model|sentencepiece\.bpe\.model|special_tokens_map\.json|"
                    r"added_tokens\.json|bpe\.codes)$")

UNCHANGED, CONVERTED, REWRITTEN, META, UNKNOWN = "UNCHANGED", "CONVERTED", "REWRITTEN", "METADATA", "UNKNOWN"
EXIT = {UNCHANGED: 0, CONVERTED: 0, META: 0, REWRITTEN: 1, UNKNOWN: 2}


class GitError(RuntimeError):
    pass


def _git(cwd, *args, token=None, timeout=300):
    auth = ["-c", f"http.extraHeader=Authorization: Bearer {token}"] if token else []
    cmd = ["git", *auth, *(["-C", cwd] if cwd else []), *args]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        err = r.stderr
        if token:
            err = err.replace(token, "***")
        raise GitError(err.strip()[-300:])
    return r.stdout


def clone_metadata(model: str, token=None) -> str:
    """Bare, blob-less clone: commits and file trees, no weights. Caller removes the directory."""
    d = tempfile.mkdtemp(prefix="su-")
    url = model if os.path.isdir(model) else f"{HUB}/{model}"
    try:
        _git(None, "clone", "-q", "--bare", "--filter=blob:none", url, d, token=token)
    except GitError:
        shutil.rmtree(d, ignore_errors=True)
        d = tempfile.mkdtemp(prefix="su-")
        _git(None, "clone", "-q", "--bare", url, d, token=token)   # server may refuse partial clone
    return d


def _state_at(repo: str, rev: str) -> dict:
    """Map path -> blob id for the repository root at `rev`. For LFS files the blob is the pointer,
    which changes exactly when the stored file changes, so no weights are fetched."""
    out = _git(repo, "ls-tree", "--full-tree", rev)
    state = {}
    for line in out.splitlines():
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) >= 3 and parts[1] == "blob":
            state[path] = parts[2]
    return state


def _fp(state, paths):
    return hashlib.sha1("|".join(f"{p}:{state[p]}" for p in paths).encode()).hexdigest()[:16] if paths else None


def _effective(state):
    """What from_pretrained loads: safetensors when present, else .bin."""
    st = sorted(p for p in state if WEIGHT_RE.match(p) and p.endswith(".safetensors"))
    bn = sorted(p for p in state if WEIGHT_RE.match(p) and p.endswith(".bin"))
    paths, fmt = (st, "safetensors") if st else (bn, "bin") if bn else ([], None)
    return fmt, _fp(state, paths), _fp(state, bn)


def compare(repo: str, base_rev: str, head_rev: str) -> dict:
    a, b = _state_at(repo, base_rev), _state_at(repo, head_rev)
    fa, wa, ba = _effective(a)
    fb, wb, bb = _effective(b)
    cfg_changed = a.get("config.json") != b.get("config.json")
    tok_changed = _fp(a, sorted(p for p in a if TOK_RE.match(p))) != \
                  _fp(b, sorted(p for p in b if TOK_RE.match(p)))
    if wa is None or wb is None:
        verdict = UNKNOWN
    elif wa == wb:
        verdict = META if (cfg_changed or tok_changed) else UNCHANGED
    elif fa != fb and ba == bb:
        # format switched while the original .bin stayed byte-identical: the conversion bot's
        # signature. The new file is a copy of the same weights.
        verdict = CONVERTED
    else:
        verdict = REWRITTEN
    return dict(verdict=verdict, base=base_rev, head=head_rev,
                weights_changed=wa != wb, format_base=fa, format_head=fb,
                bin_changed=ba != bb, config_changed=cfg_changed, tokenizer_changed=tok_changed)


def head_sha(repo: str) -> str:
    return _git(repo, "rev-parse", "HEAD").strip()


def resolve(repo: str, rev: str) -> str:
    try:
        return _git(repo, "rev-parse", rev).strip()
    except GitError:
        raise GitError(f"revision not found in this repository: {rev}")


def check(model: str, since: str | None, token=None) -> dict:
    repo = clone_metadata(model, token=token)
    try:
        head = head_sha(repo)
        if since is None:
            first = _git(repo, "rev-list", "--max-parents=0", "HEAD").split()[-1]
            base = first
            note = "no --since given; compared against the first commit"
        else:
            base = resolve(repo, since)
            note = ""
        res = compare(repo, base, head)
        res.update(model=model, head=head, note=note)
        if res["verdict"] != UNCHANGED:
            log = _git(repo, "log", "--oneline", "--no-decorate", f"{base}..{head}")
            res["commits_since"] = len([l for l in log.splitlines() if l.strip()])
        return res
    finally:
        shutil.rmtree(repo, ignore_errors=True)


EXPLAIN = {
    UNCHANGED: "No change to weights, config or tokenizer. Nothing to do.",
    CONVERTED: "Weight file format changed, but the original checkpoint is byte-identical: this is a\n"
               "  safetensors conversion, which did not alter predictions for any of the 154 such\n"
               "  models measured in the study. Safe to adopt.",
    META:      "Weights unchanged; config.json or tokenizer files changed. No model in the study\n"
               "  changed a prediction this way, but label NAMES may have changed, which breaks code\n"
               "  comparing label strings. Check id2label before adopting.",
    REWRITTEN: "The stored checkpoint was rewritten. 53 of 62 such models in the study changed\n"
               "  predictions, median 17% of inputs relabelled. Re-test before adopting this version.",
    UNKNOWN:   "Could not determine: no loadable weight files found at one of the two revisions.",
}


def _print(res: dict, show_pin=True):
    v = res["verdict"]
    print(f"{v:<10s} {res['model']}")
    if res.get("note"):
        print(f"  note: {res['note']}")
    print(f"  {EXPLAIN[v]}")
    if v != UNCHANGED:
        bits = [k.replace("_changed", "") for k in ("weights_changed", "bin_changed", "config_changed",
                                                    "tokenizer_changed") if res.get(k)]
        print(f"  changed: {', '.join(bits) or 'none'}"
              f"   format: {res['format_base']} -> {res['format_head']}"
              f"   commits since: {res.get('commits_since', '?')}")
    if show_pin:
        print(f"  current head: {res['head']}")
        print(f"  pin with:     from_pretrained(\"{res['model']}\", revision=\"{res['head']}\")")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="silent-updates",
        description="Detect whether a Hugging Face model's checkpoint was rewritten in place.")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="check one model")
    c.add_argument("model")
    c.add_argument("--since", help="the revision you pinned (commit, tag or branch)")
    c.add_argument("--json", action="store_true")

    pn = sub.add_parser("pin", help="print the current head commit, ready to paste")
    pn.add_argument("model")

    a = sub.add_parser("audit", help="check a file of 'model[ revision]' lines")
    a.add_argument("file")
    a.add_argument("--json", action="store_true")

    args = p.parse_args(argv)
    token = os.environ.get("HF_TOKEN")

    try:
        if args.cmd == "pin":
            repo = clone_metadata(args.model, token=token)
            try:
                sha = head_sha(repo)
            finally:
                shutil.rmtree(repo, ignore_errors=True)
            print(f'from_pretrained("{args.model}", revision="{sha}")')
            return 0

        if args.cmd == "check":
            res = check(args.model, args.since, token=token)
            print(json.dumps(res) if args.json else "", end="")
            if not args.json:
                _print(res)
            return EXIT[res["verdict"]]

        rows, worst = [], 0
        for line in open(args.file):
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.split()
            model, rev = parts[0], (parts[1] if len(parts) > 1 else None)
            try:
                res = check(model, rev, token=token)
            except Exception as e:
                res = dict(model=model, verdict=UNKNOWN, head="", error=f"{type(e).__name__}: {e}")
            rows.append(res)
            worst = max(worst, EXIT[res["verdict"]])
            if not args.json:
                _print(res, show_pin=res["verdict"] == REWRITTEN)
                print()
        if args.json:
            print(json.dumps(rows, indent=1))
        else:
            n = sum(r["verdict"] == REWRITTEN for r in rows)
            print(f"{len(rows)} checked, {n} with a rewritten checkpoint")
        return worst
    except GitError as e:
        print(f"git error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
