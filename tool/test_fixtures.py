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

print(f"{'scenario':<34s} {'expected':<10s} {'got':<10s} result")
fails = 0
for name, repo, base, expect in CASES:
    res = su.check(repo, base)
    got = res["verdict"]
    ok = got == expect
    fails += not ok
    print(f"{name:<34s} {expect:<10s} {got:<10s} {'pass' if ok else 'FAIL'}")

# exit codes: a rewritten checkpoint must fail CI, a conversion must not
print()
for name, repo, base, expect in CASES:
    code = su.EXIT[su.check(repo, base)["verdict"]]
    want = 1 if expect == su.REWRITTEN else 0
    if code != want:
        print(f"EXIT CODE FAIL {name}: {code} != {want}"); fails += 1
print("exit codes checked")

shutil.rmtree(ROOT, ignore_errors=True)
print(f"\n{len(CASES)} scenarios, {fails} failures")
sys.exit(1 if fails else 0)
