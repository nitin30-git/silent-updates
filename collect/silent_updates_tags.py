# Silent Updates - experiment 4: is in-place rewriting a text-classification quirk or Hub-wide?
# ----------------------------------------------------------------------------------------------
# Kaggle or Colab. Internet ON, secret HF_TOKEN. NO GPU. About 60-90 minutes.
#
# Cell 1: !pip -q install -U "huggingface_hub<1.0"
# Cell 2: this file.
#
# The main study covers text classification only, which is the paper's weakest external-validity
# claim. This repeats just the history stage (no inference, no weights downloaded) across several
# pipeline tags, so the prevalence result can be stated for the Hub rather than for one task.
#
# Everything here is the measurement code from the main study, unchanged, applied per tag.
# ----------------------------------------------------------------------------------------------
import os, re, json, time, shutil, hashlib, subprocess, tempfile, traceback
import concurrent.futures as cf
from datetime import datetime, timedelta

SCRATCH = "/tmp/su_tags"; os.makedirs(SCRATCH, exist_ok=True)
os.environ.update(HF_HUB_CACHE=f"{SCRATCH}/hub", HF_HUB_DISABLE_XET="1",
                  HF_HUB_DISABLE_PROGRESS_BARS="1", HF_HUB_DISABLE_TELEMETRY="1",
                  GIT_TERMINAL_PROMPT="0", GIT_LFS_SKIP_SMUDGE="1")

PLATFORM = "kaggle" if os.path.exists("/kaggle/working") else "colab" if os.path.exists("/content") else "local"
TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN and PLATFORM == "kaggle":
    from kaggle_secrets import UserSecretsClient; TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
elif not TOKEN and PLATFORM == "colab":
    from google.colab import userdata; TOKEN = userdata.get("HF_TOKEN")
assert TOKEN, "HF_TOKEN not found"
if PLATFORM == "kaggle": os.chdir("/kaggle/working")

from huggingface_hub import login, HfApi
login(TOKEN); api = HfApi()

TAGS = ["text-classification",        # repeat of the main study, as an internal check
        "token-classification",
        "image-classification",
        "sentence-similarity",
        "fill-mask",
        "automatic-speech-recognition",
        "object-detection"]
N_PER_TAG, GRACE, WORKERS = 250, 7, 6
OUT = "su_tags"; os.makedirs(OUT, exist_ok=True)
SKIP = ("internal-testing", "tiny-random", "onnx", "gguf", "openvino", "hf-internal")
BUDGET_S = 95 * 60
t_start = time.time()
print(f"tags: {len(TAGS)} x {N_PER_TAG} models, no GPU, budget {BUDGET_S//60} min")

# ---------------------------------------------------------------- measurement code (as published)
WEIGHT_RE = re.compile(r"^(model|pytorch_model)(-\d{5}-of-\d{5})?\.(safetensors|bin)$")
TOK_RE = re.compile(r"^(tokenizer\.json|tokenizer_config\.json|vocab\.(txt|json)|merges\.txt|spiece\.model|"
                    r"sentencepiece\.bpe\.model|special_tokens_map\.json|added_tokens\.json|bpe\.codes|"
                    r"preprocessor_config\.json)$")
AUTH = ["-c", f"http.extraHeader=Authorization: Bearer {TOKEN}"]
ts = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))

def git(d, *a, timeout=300):
    r = subprocess.run(["git", *AUTH, *(["-C", d] if d else []), *a], capture_output=True, text=True, timeout=timeout)
    if r.returncode: raise RuntimeError(r.stderr.replace(TOKEN, "***")[-200:])
    return r.stdout

def fp(state, paths):
    return hashlib.sha1("|".join(f"{p}:{state[p]}" for p in paths).encode()).hexdigest()[:16] if paths else None

def eff_weights(state):
    st = sorted(p for p in state if WEIGHT_RE.match(p) and p.endswith(".safetensors"))
    bn = sorted(p for p in state if WEIGHT_RE.match(p) and p.endswith(".bin"))
    ps, fmt = (st, "safetensors") if st else (bn, "bin") if bn else ([], None)
    return fmt, fp(state, ps), ps

def parse_log(txt):
    commits, cur = [], None
    for line in txt.splitlines():
        if line.startswith("@@"):
            sha, t, author, *title = line[2:].split("\t")
            cur = dict(sha=sha, t=t, author=author, title="\t".join(title), ch=[]); commits.append(cur)
        elif line.startswith(":") and cur is not None:
            meta, path = line.split("\t", 1)
            _, _, _, new_blob, status = meta.split()
            cur["ch"].append((status[0], path, new_blob))
    return commits

def timeline(commits):
    state, events, prev = {}, [], None
    for c in commits:
        for st_, path, blob in c["ch"]:
            if st_ == "D": state.pop(path, None)
            else: state[path] = blob
        fmt, wfp, _ = eff_weights(state)
        binfp = fp(state, sorted(p for p in state if WEIGHT_RE.match(p) and p.endswith(".bin")))
        tok = fp(state, sorted(p for p in state if TOK_RE.match(p)))
        cur = (wfp, fmt, state.get("config.json"), tok, binfp)
        if wfp is not None and cur != prev:
            events.append(dict(sha=c["sha"], t=c["t"], w=wfp, fmt=fmt, cfg=cur[2], tok=tok, wbin=binfp,
                               readme=any(p == "README.md" for _, p, _ in c["ch"])))
            prev = cur
    return events

def history(mid):
    d = tempfile.mkdtemp(dir=SCRATCH)
    try:
        url = f"https://huggingface.co/{mid}"
        try: git(None, "clone", "-q", "--bare", "--filter=blob:none", url, d)
        except Exception:
            shutil.rmtree(d, ignore_errors=True); d = tempfile.mkdtemp(dir=SCRATCH)
            git(None, "clone", "-q", "--bare", url, d)
        commits = parse_log(git(d, "log", "--reverse", "--first-parent", "-m", "--raw", "--no-renames",
                                "--no-abbrev", "--format=@@%H%x09%cI%x09%an%x09%s", "HEAD"))
        ev = timeline(commits)
        return dict(id=mid, events=ev, head=commits[-1]["sha"] if commits else None,
                    head_t=commits[-1]["t"] if commits else None)
    finally:
        shutil.rmtree(d, ignore_errors=True)

def classify(x, grace=GRACE):
    ev = x.get("events") or []
    if not ev or not x.get("head"): return None
    cut = ts(ev[0]["t"]) + timedelta(days=grace)
    rel = [e for e in ev if ts(e["t"]) <= cut][-1]; last = ev[-1]
    wdiff = rel["w"] != last["w"]
    same_fmt = rel["fmt"] == last["fmt"]
    bindiff = rel["wbin"] != last["wbin"]
    if not wdiff:
        kind = "meta" if (rel["cfg"] != last["cfg"] or rel["tok"] != last["tok"]) else "none"
    elif same_fmt: kind = "rewrite"
    else: kind = "rewrite" if bindiff else "convert"
    return dict(id=x["id"], kind=kind, weights_differ=wdiff,
                cfg=rel["cfg"] != last["cfg"], tok=rel["tok"] != last["tok"])

# ---------------------------------------------------------------- run
RES_PATH = f"{OUT}/tags.json"
RES = json.load(open(RES_PATH)) if os.path.exists(RES_PATH) else {}

for tag in TAGS:
    if tag in RES and RES[tag].get("done"):
        print(f"\n[{tag}] already done, skipping"); continue
    if time.time() - t_start > BUDGET_S:
        print("\nSTOPPING: budget reached. Re-run to continue with the remaining tags."); break
    print(f"\n[{tag}] listing models")
    models = []
    try:
        for m in api.list_models(pipeline_tag=tag, sort="downloads", limit=N_PER_TAG + 60):
            if any(s in m.id.lower() for s in SKIP): continue
            models.append(dict(id=m.id, downloads=m.downloads or 0))
            if len(models) >= N_PER_TAG: break
    except Exception as e:
        print(f"  listing failed: {type(e).__name__}: {e}"); RES[tag] = dict(error=str(e)[:200]); continue
    print(f"[{tag}] fetching {len(models)} histories")
    cls, errs, t0 = {}, 0, time.time()
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(history, m["id"]): m["id"] for m in models}
        for i, f in enumerate(cf.as_completed(futs)):
            try:
                c = classify(f.result())
                if c: cls[c["id"]] = c
            except Exception: errs += 1
            if i and i % 100 == 0: print(f"    {i}/{len(models)} ({time.time()-t0:.0f}s)")
    n = len(cls)
    cnt = {k: sum(c["kind"] == k for c in cls.values()) for k in ("rewrite", "convert", "meta", "none")}
    any_ch = n - cnt["none"]
    RES[tag] = dict(done=True, listed=len(models), with_weights=n, errors=errs, counts=cnt,
                    any_change=any_ch,
                    rewrite_pct=round(100 * cnt["rewrite"] / n, 1) if n else None,
                    any_pct=round(100 * any_ch / n, 1) if n else None,
                    dl=sum(m["downloads"] for m in models))
    json.dump(RES, open(RES_PATH, "w"), indent=1)
    print(f"[{tag}] with weights {n} | rewritten {cnt['rewrite']} ({RES[tag]['rewrite_pct']}%) | "
          f"converted {cnt['convert']} | metadata-only {cnt['meta']} | any change {any_ch} ({RES[tag]['any_pct']}%)")
    shutil.rmtree(SCRATCH, ignore_errors=True); os.makedirs(SCRATCH, exist_ok=True)

# ---------------------------------------------------------------- summary
print("\n" + "=" * 86)
print(f"{'pipeline tag':<30s} {'n':>5s} {'rewritten':>12s} {'converted':>10s} {'meta':>6s} {'any change':>12s}")
print("=" * 86)
for tag in TAGS:
    r = RES.get(tag)
    if not r or not r.get("done"): continue
    c = r["counts"]
    print(f"{tag:<30s} {r['with_weights']:5d} {c['rewrite']:6d} {r['rewrite_pct']:5.1f}% "
          f"{c['convert']:10d} {c['meta']:6d} {r['any_change']:6d} {r['any_pct']:5.1f}%")
done = [RES[t] for t in TAGS if RES.get(t, {}).get("done")]
if done:
    tot_n = sum(r["with_weights"] for r in done); tot_rw = sum(r["counts"]["rewrite"] for r in done)
    print("=" * 86)
    print(f"{'ALL TAGS':<30s} {tot_n:5d} {tot_rw:6d} {100*tot_rw/tot_n:5.1f}%")

print("\n---------- RESULTS JSON (copy this if the output tab fails) ----------")
print(json.dumps(RES))
try:
    shutil.make_archive("su_tags_results", "zip", ".", OUT); print("\nsaved su_tags_results.zip")
except Exception as e: print("zip failed:", e)
