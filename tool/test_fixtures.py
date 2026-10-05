#!/usr/bin/env python3
"""Build local git repositories that reproduce each Hub scenario and check the verdicts.
No network: the tool clones from a local path exactly as it would from the Hub."""
import os, shutil, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import silent_updates as su

ROOT = tempfile.mkdtemp(prefix="su-fixtures-")

def run(d, *a):
    subprocess.run(["git", "-C", d, *a], check=True, capture_output=True)

def mkrepo(name):
    d = os.path.join(ROOT, name); os.makedirs(d)
    run(d, "init", "-q", "-b", "main")
    run(d, "config", "user.email", "t@t.t"); run(d, "config", "user.name", "t")
    return d

def write(d, path, content):
    with open(os.path.join(d, path), "w") as f: f.write(content)

def commit(d, msg):
    run(d, "add", "-A"); run(d, "commit", "-q", "-m", msg)
    return subprocess.run(["git", "-C", d, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()

CASES = []

# 1. nothing happens after release
d = mkrepo("unchanged")
write(d, "config.json", '{"model_type":"bert","id2label":{"0":"NEG","1":"POS"}}')
write(d, "pytorch_model.bin", "WEIGHTS-v1")
write(d, "vocab.txt", "a\nb\n")
base = commit(d, "release")
write(d, "README.md", "docs")
commit(d, "docs only")
CASES.append(("unchanged", d, base, su.UNCHANGED))

# 2. the conversion bot adds safetensors; the original .bin is untouched
d = mkrepo("converted")
write(d, "config.json", '{"model_type":"bert"}')
write(d, "pytorch_model.bin", "WEIGHTS-v1")
base = commit(d, "release")
write(d, "model.safetensors", "WEIGHTS-v1-as-safetensors")
commit(d, "Adding `safetensors` variant of this model")
CASES.append(("converted (bot)", d, base, su.CONVERTED))

# 3. the maintainer overwrites the weights in the same format
d = mkrepo("rewritten")
write(d, "config.json", '{"model_type":"bert"}')
write(d, "model.safetensors", "WEIGHTS-v1")
base = commit(d, "release")
write(d, "model.safetensors", "WEIGHTS-v2-retrained")
commit(d, "update model")
CASES.append(("rewritten in place", d, base, su.REWRITTEN))

# 4. format switch AND the .bin changed: not a conversion of what was published
d = mkrepo("rewritten_switch")
write(d, "config.json", '{"model_type":"bert"}')
write(d, "pytorch_model.bin", "WEIGHTS-v1")
base = commit(d, "release")
write(d, "pytorch_model.bin", "WEIGHTS-v2")
write(d, "model.safetensors", "WEIGHTS-v2-as-safetensors")
commit(d, "retrain and convert")
CASES.append(("rewrite + format switch", d, base, su.REWRITTEN))

# 5. labels renamed, weights identical
d = mkrepo("metadata")
write(d, "config.json", '{"model_type":"bert","id2label":{"0":"LABEL_0","1":"LABEL_1"}}')
write(d, "model.safetensors", "WEIGHTS-v1")
base = commit(d, "release")
write(d, "config.json", '{"model_type":"bert","id2label":{"0":"negative","1":"positive"}}')
commit(d, "rename labels")
CASES.append(("labels renamed only", d, base, su.META))

# 6. sharded weights, one shard rewritten
d = mkrepo("sharded")
write(d, "config.json", '{"model_type":"bert"}')
write(d, "model-00001-of-00002.safetensors", "SHARD-1")
write(d, "model-00002-of-00002.safetensors", "SHARD-2")
base = commit(d, "release")
write(d, "model-00002-of-00002.safetensors", "SHARD-2-v2")
commit(d, "fix shard 2")
CASES.append(("sharded, one shard rewritten", d, base, su.REWRITTEN))

# 7. training_args.bin churn must NOT count as a weight change
d = mkrepo("trainingargs")
write(d, "config.json", '{"model_type":"bert"}')
write(d, "model.safetensors", "WEIGHTS-v1")
write(d, "training_args.bin", "ARGS-v1")
base = commit(d, "release")
write(d, "training_args.bin", "ARGS-v2")
write(d, "optimizer.pt", "OPT")
commit(d, "upload training state")
CASES.append(("training_args churn only", d, base, su.UNCHANGED))

# 8. tokenizer changed, weights identical
d = mkrepo("tokenizer")
write(d, "config.json", '{"model_type":"bert"}')
write(d, "model.safetensors", "WEIGHTS-v1")
write(d, "tokenizer.json", "TOK-v1")
base = commit(d, "release")
write(d, "tokenizer.json", "TOK-v2")
commit(d, "fix tokenizer")
CASES.append(("tokenizer only", d, base, su.META))

# 9. default release state: first commit holds no weights, and an upload fix-up lands inside the grace period
def commit_at(d, msg, date):
    run(d, "add", "-A")
    env = dict(os.environ, GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date)
    subprocess.run(["git", "-C", d, "commit", "-q", "-m", msg], check=True, capture_output=True, env=env)

d = mkrepo("default_release")
write(d, ".gitattributes", "*.bin filter=lfs")
commit_at(d, "initial commit", "2023-01-01T10:00:00+00:00")
write(d, "config.json", '{"model_type":"bert"}'); write(d, "pytorch_model.bin", "WEIGHTS-v1")
commit_at(d, "add model", "2023-01-02T10:00:00+00:00")
write(d, "pytorch_model.bin", "WEIGHTS-v1-fixed")             # inside the 7-day grace period: part of the release
commit_at(d, "re-upload", "2023-01-04T10:00:00+00:00")
write(d, "pytorch_model.bin", "WEIGHTS-v2-retrained")         # months later: a real rewrite
commit_at(d, "retrained", "2023-06-01T10:00:00+00:00")
DEFAULT_CASES = [("default release, retrained later", d, su.REWRITTEN)]

d = mkrepo("default_release_unchanged")
write(d, ".gitattributes", "*.bin filter=lfs")
commit_at(d, "initial commit", "2023-01-01T10:00:00+00:00")
write(d, "config.json", '{"model_type":"bert"}'); write(d, "pytorch_model.bin", "WEIGHTS-v1")
commit_at(d, "add model", "2023-01-02T10:00:00+00:00")
write(d, "pytorch_model.bin", "WEIGHTS-v1-fixed")
commit_at(d, "re-upload", "2023-01-03T10:00:00+00:00")
DEFAULT_CASES.append(("default release, only upload fix-up", d, su.UNCHANGED))

d = mkrepo("default_release_converted")
write(d, ".gitattributes", "*.bin filter=lfs")
commit_at(d, "initial commit", "2023-01-01T10:00:00+00:00")
write(d, "config.json", '{"model_type":"bert"}'); write(d, "pytorch_model.bin", "WEIGHTS-v1")
commit_at(d, "add model", "2023-01-02T10:00:00+00:00")
write(d, "model.safetensors", "WEIGHTS-v1-as-safetensors")
commit_at(d, "Adding `safetensors` variant of this model", "2024-03-01T10:00:00+00:00")
DEFAULT_CASES.append(("default release, bot conversion", d, su.CONVERTED))

# 10. conversion that also touches config.json must say so
d = mkrepo("converted_cfg")
write(d, "config.json", '{"model_type":"bert","id2label":{"0":"LABEL_0"}}')
write(d, "pytorch_model.bin", "WEIGHTS-v1")
base = commit(d, "release")
write(d, "model.safetensors", "WEIGHTS-v1-as-safetensors")
write(d, "config.json", '{"model_type":"bert","id2label":{"0":"negative"}}')
commit(d, "Adding `safetensors` variant of this model")
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    su._print(su.check(d, base))
warn_ok = "WARNING: config.json also changed" in buf.getvalue()
CASES.append(("conversion + config edit", d, base, su.CONVERTED))

print(f"{'scenario':<34s} {'expected':<10s} {'got':<10s} result")
fails = 0
for name, repo, base, expect in CASES:
    res = su.check(repo, base)
    got = res["verdict"]
    ok = got == expect
    fails += not ok
    print(f"{name:<34s} {expect:<10s} {got:<10s} {'pass' if ok else 'FAIL'}")

for name, repo, expect in DEFAULT_CASES:
    got = su.check(repo, None)["verdict"]
    ok = got == expect
    fails += not ok
    print(f"{name:<34s} {expect:<10s} {got:<10s} {'pass' if ok else 'FAIL'}")

print(f"{'config warning shown':<34s} {'yes':<10s} {'yes' if warn_ok else 'no':<10s} {'pass' if warn_ok else 'FAIL'}"); fails += not warn_ok

# exit codes: a rewritten checkpoint must fail CI, a conversion must not
print()
for name, repo, base, expect in CASES:
    code = su.EXIT[su.check(repo, base)["verdict"]]
    want = 1 if expect == su.REWRITTEN else 0
    if code != want:
        print(f"EXIT CODE FAIL {name}: {code} != {want}"); fails += 1
print("exit codes checked")

shutil.rmtree(ROOT, ignore_errors=True)
print(f"\n{len(CASES) + len(DEFAULT_CASES)} scenarios, {fails} failures")
sys.exit(1 if fails else 0)
