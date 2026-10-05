# Silent Updates on Hugging Face - full study run (final)
# ------------------------------------------------------------------------------------------------
# Kaggle (recommended) or Colab. GPU (T4) + Internet ON + secret HF_TOKEN (read token).
#
# Cell 1:  !pip -q install -U "transformers>=4.46,<5" "huggingface_hub<1.0" datasets safetensors sentencepiece protobuf tiktoken fugashi unidic-lite
#          (transformers 4.x on purpose: 5.x can no longer load many 2019-2022 checkpoints, which are exactly the ones we study)
# Cell 2:  (paste this whole file)
#
# MODES (automatic on Kaggle):
#   * Interactive session  -> SMOKE TEST: 6 hand-picked models, ~5 min, prints versions + full errors.
#   * Save & Run All (Commit) -> FULL RUN: ~1000 models, ~4 h, output saved in the version's Output tab.
#   Override: set os.environ["SU_MODE"] = "full" or "smoke" in a cell before running.
#
# What it measures (per model): RELEASE = repo state GRACE_DAYS after the first weights were uploaded;
# LATEST = HEAD today. It records every change to loadable weights / config.json / tokenizer files,
# runs RELEASE and LATEST on the same 2,872 texts, saves raw probabilities, and probes the Spaces
# that use each changed model (pinned revision? built before the change?).
#
# Safety: never runs remote code; never uses the background safetensors "auto conversion" (which calls a
# Space with your token); all Hugging Face caches live in one scratch dir that is wiped after every model;
# disk and time watchdogs stop the run cleanly so the output is always saved.
# ------------------------------------------------------------------------------------------------
import os, re, sys, json, time, shutil, random, hashlib, subprocess, tempfile, traceback, collections
import concurrent.futures as cf
from datetime import datetime, timedelta

# ---------- environment (must be set BEFORE importing huggingface_hub / transformers) ----------
SCRATCH = "/tmp/su_scratch"                      # every download lands here; wiped after each model
os.makedirs(SCRATCH, exist_ok=True)
os.environ.update(
    HF_HUB_CACHE=f"{SCRATCH}/hub", HF_XET_CACHE=f"{SCRATCH}/xet", HF_HUB_DISABLE_XET="1",
    HF_HUB_DISABLE_PROGRESS_BARS="1", HF_HUB_DISABLE_TELEMETRY="1", TRANSFORMERS_VERBOSITY="error",
    TQDM_DISABLE="1", TOKENIZERS_PARALLELISM="false", DISABLE_SAFETENSORS_CONVERSION="1",
    TORCHINDUCTOR_CACHE_DIR=f"{SCRATCH}/inductor", GIT_TERMINAL_PROMPT="0", GIT_LFS_SKIP_SMUDGE="1",
    TORCHDYNAMO_DISABLE="1")   # no torch.compile (some models, e.g. ModernBERT, compile on GPU: slow, and can stall)

PLATFORM = "kaggle" if os.path.exists("/kaggle/working") else "colab" if os.path.exists("/content") else "local"
TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN and PLATFORM == "kaggle":
    from kaggle_secrets import UserSecretsClient; TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
elif not TOKEN and PLATFORM == "colab":
    from google.colab import userdata; TOKEN = userdata.get("HF_TOKEN")
assert TOKEN, "HF_TOKEN not found (Kaggle: Add-ons > Secrets, toggle it ON for this notebook)"
if PLATFORM == "kaggle": os.chdir("/kaggle/working")

MODE = os.environ.get("SU_MODE") or ("smoke" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") == "Interactive" else "full")
SMOKE_MODELS = ["distilbert/distilbert-base-uncased-finetuned-sst-2-english",   # old .bin at release
                "yiyanghkust/finbert-tone",                                      # failed in the last run
                "pysentimiento/robertuito-sentiment-analysis",                   # real behaviour change
                "tabularisai/multilingual-sentiment-analysis",                   # labels renamed
                "microsoft/MiniLM-L12-H384-uncased",                             # random head -> must be flagged
                "cardiffnlp/twitter-roberta-base-irony"]                         # name-only change
N_MODELS, GRACE_DAYS, MAX_GB, WORKERS = 1000, 7, 1.5, 6
INFER_BUDGET_S = int(6.5 * 3600) if MODE == "full" else 900   # checked between models; each model is hard-limited below
PER_MODEL_TIMEOUT_S = 900                                      # one model (download + 2 loads + 2,872 texts x2) may never take longer
OUT = "su_v5" if MODE == "full" else "su_smoke"
print(f"MODE={MODE}  PLATFORM={PLATFORM}  OUT={OUT}")

try:                                                   # catches "pip upgraded, but the old version is still in memory"
    import huggingface_hub, huggingface_hub._snapshot_download, transformers   # noqa: F401
    print(f"huggingface_hub {huggingface_hub.__version__} | transformers {transformers.__version__}")
except ImportError as e:
    raise SystemExit(f"LIBRARY VERSION MIX IN MEMORY ({e}).\n-> Menu: Run > Restart session (restart the kernel), then run THIS cell again. No need to re-run pip.")
from huggingface_hub import login, HfApi, hf_hub_download
login(TOKEN); api = HfApi()

if MODE == "full" and PLATFORM == "kaggle" and not os.path.exists(OUT):     # resume from an attached earlier output
    for root, dirs, _ in os.walk("/kaggle/input"):
        if OUT in dirs: shutil.copytree(os.path.join(root, OUT), OUT); print("resumed from", root); break
os.makedirs(f"{OUT}/preds", exist_ok=True)

# ---------- helpers ----------
def retry(fn, *a, tries=6, **k):
    for i in range(tries):
        try: return fn(*a, **k)
        except Exception as e:
            if any(x in str(e) for x in ("429", "Too Many", "rate limit", "502", "503", "504", "timed out", "Connection", "Temporary failure")):
                time.sleep(15 * (i + 1)); continue
            raise
    raise RuntimeError("gave up after retries")

def free_gb(path="/"): return shutil.disk_usage(path).free / 1e9
def du_gb(path):
    tot = 0
    for r, _, fs in os.walk(path):
        for f in fs:
            try: tot += os.path.getsize(os.path.join(r, f))
            except OSError: pass
    return tot / 1e9
def wipe_scratch():
    for d in os.listdir(SCRATCH): shutil.rmtree(os.path.join(SCRATCH, d), ignore_errors=True)
    for d in os.listdir("/tmp"):
        if d.startswith("torchinductor"): shutil.rmtree(os.path.join("/tmp", d), ignore_errors=True)
    shutil.rmtree(os.path.expanduser("~/.cache/huggingface/hub"), ignore_errors=True)   # just in case something ignored HF_HUB_CACHE
    shutil.rmtree(os.path.expanduser("~/.cache/huggingface/xet"), ignore_errors=True)
def disk_report(tag=""):
    print(f"[disk {tag}] free /: {free_gb():.1f} GB | free work: {free_gb('.'):.1f} GB | scratch: {du_gb(SCRATCH):.2f} GB | out: {du_gb(OUT):.2f} GB")
def ts(s): return datetime.fromisoformat(s.replace("Z", "+00:00"))

def save_zip():
    try:
        shutil.make_archive(f"{OUT}_results", "zip", ".", OUT); print(f"saved {OUT}_results.zip")
    except Exception as e: print("zip failed:", e)

# ---------- backup to a private Hugging Face dataset repo ----------
# Kaggle keeps a version's output only if the whole commit finalises; runs have been lost that way.
# This pushes results to the Hub as they are produced, so they survive regardless. Never fatal.
BACKUP_REPO = None
def backup_init():
    global BACKUP_REPO
    try:
        BACKUP_REPO = f'{api.whoami()["name"]}/silent-updates-results'
        api.create_repo(BACKUP_REPO, repo_type="dataset", private=True, exist_ok=True)
        print("backup repo:", BACKUP_REPO)
    except Exception as e:
        BACKUP_REPO = None
        print(f"BACKUP DISABLED ({type(e).__name__}: {str(e)[:120]}). Needs a WRITE token; "
              "results would then live only in Kaggle's output.")

def backup(tag, preds=False):
    """Push OUT/ to the Hub. preds=False skips the per-model .npz files (bulky; only needed at the end)."""
    if not BACKUP_REPO: return
    try:
        t = time.time()
        api.upload_folder(folder_path=OUT, path_in_repo=OUT, repo_type="dataset", repo_id=BACKUP_REPO,
                          commit_message=tag, ignore_patterns=None if preds else ["preds/*"])
        print(f"  [backup {tag}] pushed in {time.time() - t:.0f}s")
    except Exception as e:
        print(f"  [backup {tag}] FAILED ({type(e).__name__}: {str(e)[:120]}) - continuing")

backup_init()
disk_report("start")

# =================================================================================================
# A) model list
# =================================================================================================
SKIP = ("internal-testing", "tiny-random", "onnx", "gguf", "openvino", "hf-internal")
if os.path.exists(f"{OUT}/models.json"):
    MODELS = json.load(open(f"{OUT}/models.json"))
elif MODE == "smoke":
    MODELS = [dict(id=m, downloads=0, likes=0, created="", library=None) for m in SMOKE_MODELS]
else:
    MODELS = []
    for m in api.list_models(pipeline_tag="text-classification", sort="downloads", limit=N_MODELS + 60):
        if any(s in m.id.lower() for s in SKIP): continue
        MODELS.append(dict(id=m.id, downloads=m.downloads or 0, likes=m.likes or 0,
                           created=str(getattr(m, "created_at", "")), library=getattr(m, "library_name", None)))
        if len(MODELS) >= N_MODELS: break
json.dump(MODELS, open(f"{OUT}/models.json", "w"))
print("models:", len(MODELS))

# =================================================================================================
# B) full change history of loadable files via bare git clone (metadata only, no weights)
# =================================================================================================
WEIGHT_RE = re.compile(r"^(model|pytorch_model)(-\d{5}-of-\d{5})?\.(safetensors|bin)$")
TOK_RE    = re.compile(r"^(tokenizer\.json|tokenizer_config\.json|vocab\.(txt|json)|merges\.txt|spiece\.model|"
                       r"sentencepiece\.bpe\.model|special_tokens_map\.json|added_tokens\.json|bpe\.codes)$")
AUTH = ["-c", f"http.extraHeader=Authorization: Bearer {TOKEN}"]

def git(d, *args, timeout=300):
    r = subprocess.run(["git", *AUTH, *(["-C", d] if d else []), *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode: raise RuntimeError(r.stderr.replace(TOKEN, "***")[-200:])
    return r.stdout

def fp(state, paths):
    return hashlib.sha1("|".join(f"{p}:{state[p]}" for p in paths).encode()).hexdigest()[:16] if paths else None

def eff_weights(state):
    """What from_pretrained loads: safetensors if present, else .bin -> (format, fingerprint, paths)."""
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

def timeline_from_commits(commits):
    state, events, prev = {}, [], None
    for c in commits:
        for st_, path, blob in c["ch"]:
            if st_ == "D": state.pop(path, None)
            else: state[path] = blob
        fmt, wfp, _ = eff_weights(state)
        bin_fp = fp(state, sorted(p for p in state if WEIGHT_RE.match(p) and p.endswith(".bin")))
        tok = fp(state, sorted(p for p in state if TOK_RE.match(p)))
        cur = (wfp, fmt, state.get("config.json"), tok, bin_fp)
        if wfp is not None and cur != prev:
            events.append(dict(sha=c["sha"], t=c["t"], author=c["author"], title=c["title"][:120],
                               w=wfp, fmt=fmt, cfg=cur[2], tok=tok, wbin=bin_fp,
                               readme=any(p == "README.md" for _, p, _ in c["ch"])))
            prev = cur
    return events, state

def history(mid):
    d = tempfile.mkdtemp(dir=SCRATCH)
    try:
        url = f"https://huggingface.co/{mid}"
        try: git(None, "clone", "-q", "--bare", "--filter=blob:none", url, d)
        except Exception:
            shutil.rmtree(d, ignore_errors=True); d = tempfile.mkdtemp(dir=SCRATCH)
            git(None, "clone", "-q", "--bare", url, d)
        commits = parse_log(git(d, "log", "--reverse", "--first-parent", "-m", "--raw", "--no-renames", "--no-abbrev",
                                "--format=@@%H%x09%cI%x09%an%x09%s", "HEAD"))
        events, state = timeline_from_commits(commits)
        refs = [r for r in git(d, "for-each-ref", "--format=%(refname)").split() if r not in ("refs/heads/main", "HEAD")]
        gb = 0.0
        for p in eff_weights(state)[2]:                       # size from the tiny LFS pointer blob
            try:
                m = re.search(r"size (\d+)", git(d, "cat-file", "-p", state[p], timeout=60))
                gb += int(m.group(1)) / 1e9 if m else 0
            except Exception: pass
        return dict(id=mid, n_commits=len(commits), head=commits[-1]["sha"] if commits else None,
                    head_t=commits[-1]["t"] if commits else None, first_commit_t=commits[0]["t"] if commits else None,
                    events=events, refs=refs, gb=round(gb, 3))
    finally:
        shutil.rmtree(d, ignore_errors=True)

TL_PATH = f"{OUT}/timelines.jsonl"; TL = {}
if os.path.exists(TL_PATH):
    for l in open(TL_PATH):
        x = json.loads(l); TL[x["id"]] = x
todo = [m["id"] for m in MODELS if m["id"] not in TL or "error" in TL[m["id"]]]
print("histories to fetch:", len(todo))
with cf.ThreadPoolExecutor(WORKERS) as ex, open(TL_PATH, "a") as fh:
    futs = {ex.submit(retry, history, mid, tries=3): mid for mid in todo}
    for i, f in enumerate(cf.as_completed(futs)):
        mid = futs[f]
        try: x = f.result()
        except Exception as e: x = dict(id=mid, error=str(e)[:200])
        TL[mid] = x; fh.write(json.dumps(x) + "\n"); fh.flush()
        if i % 50 == 0: print(f"  {i}/{len(todo)}")
print("histories:", len(TL), "| errors:", sum("error" in x for x in TL.values()))
backup("histories")
wipe_scratch()

# =================================================================================================
# C) classify: RELEASE (first weights + GRACE_DAYS) vs LATEST
# =================================================================================================
def classify(x, grace=GRACE_DAYS):
    ev = x.get("events") or []
    if not ev or not x.get("head"): return None
    cut = ts(ev[0]["t"]) + timedelta(days=grace)
    rel = [e for e in ev if ts(e["t"]) <= cut][-1]
    post = [e for e in ev if ts(e["t"]) > cut]
    last = ev[-1]
    w_changes, prev = [], rel
    for e in post:
        if e["w"] != prev["w"]:
            w_changes.append(dict(t=e["t"], title=e["title"], author=e["author"], readme=e["readme"],
                                  kind="overwrite" if e["fmt"] == prev["fmt"] else f"format:{prev['fmt']}->{e['fmt']}"))
        prev = e
    return dict(id=x["id"], rev_R=rel["sha"], t_R=rel["t"], rev_L=x["head"], t_L=x["head_t"], t_first=ev[0]["t"],
                weights_differ=rel["w"] != last["w"], same_format=rel["fmt"] == last["fmt"],
                bin_differs=rel["wbin"] != last["wbin"], config_differs=rel["cfg"] != last["cfg"], tok_differs=rel["tok"] != last["tok"],
                n_post_weight_changes=len(w_changes), w_changes=w_changes,
                first_post_change=post[0]["t"] if post else None, last_change=last["t"], refs=x.get("refs", []), gb=x.get("gb", 0))

CL = {m: c for m, x in TL.items() if "error" not in x and (c := classify(x))}
json.dump(CL, open(f"{OUT}/classified.json", "w"))
n = max(len(CL), 1)
cand = [c for c in CL.values() if c["weights_differ"] or c["config_differs"] or c["tok_differs"]]
print(f"models with weights: {len(CL)}")
print(f"  release-vs-latest weights differ: {sum(c['weights_differ'] for c in CL.values())} ({100*sum(c['weights_differ'] for c in CL.values())/n:.1f}%)"
      f" | same-format overwrite: {sum(c['weights_differ'] and c['same_format'] for c in CL.values())}")
print(f"  config.json differs: {sum(c['config_differs'] for c in CL.values())} | tokenizer differs: {sum(c['tok_differs'] for c in CL.values())}")
print(f"  inference candidates: {len(cand)}")

# =================================================================================================
# E) Spaces that load the changed models: pinned revision? built before the change? still running?
# =================================================================================================
SP_PATH = f"{OUT}/spaces_v5.json"
SP = json.load(open(SP_PATH)) if os.path.exists(SP_PATH) else {}
HEX40 = re.compile(r"\b[0-9a-f]{40}\b")
MAX_SPACES = 50 if MODE == "full" else 5

def probe_space(sid, mid):
    info = retry(api.space_info, sid)
    files = [f for f in retry(api.list_repo_files, sid, repo_type="space") if f.endswith(".py")][:8]
    code = ""
    for f in files:
        try: code += open(retry(hf_hub_download, sid, f, repo_type="space"), encoding="utf-8", errors="ignore").read() + "\n"
        except Exception: pass
    lines = code.splitlines(); hits = [i for i, l in enumerate(lines) if mid in l]
    ctx = [" ".join(lines[max(0, i - 3):i + 4]) for i in hits]
    rt = getattr(info, "runtime", None)
    return dict(space=sid, found_in_code=bool(hits), pinned=any("revision" in c or HEX40.search(c) for c in ctx),
                snippet=(lines[hits[0]].strip()[:200] if hits else ""), created=str(getattr(info, "created_at", "")),
                last_modified=str(getattr(info, "last_modified", "")), likes=getattr(info, "likes", 0) or 0,
                sdk=getattr(info, "sdk", None), stage=str(getattr(rt, "stage", "")) if rt else "")

try:
    changed_ids = [c["id"] for c in cand if c["weights_differ"] or c["config_differs"] or c["tok_differs"]]
    for k, mid in enumerate(changed_ids):
        if mid in SP: continue
        try: spaces = [s.id for s in retry(lambda: list(api.list_spaces(models=mid, limit=MAX_SPACES)))]
        except Exception as e: SP[mid] = [dict(error=type(e).__name__)]; continue
        out = []
        with cf.ThreadPoolExecutor(WORKERS) as ex:
            for f in cf.as_completed([ex.submit(probe_space, s, mid) for s in spaces]):
                try: out.append(f.result())
                except Exception as e: out.append(dict(error=type(e).__name__))
        SP[mid] = out; json.dump(SP, open(SP_PATH, "w"))
        if k % 10 == 0: print(f"  spaces {k}/{len(changed_ids)}"); wipe_scratch()
    allsp = [s for v in SP.values() for s in v if "error" not in s]; used = [s for s in allsp if s["found_in_code"]]
    print(f"spaces: {len(allsp)} | model id in code: {len(used)} | pinned: {sum(s['pinned'] for s in used)}")
    random.seed(0)
    pool = [(m, s) for m, v in SP.items() for s in v if "error" not in s]
    with open(f"{OUT}/hand_validation.csv", "w") as fh:
        fh.write("space_url,model,auto_found,auto_pinned,your_found,your_pinned\n")
        for m, s in random.sample(pool, min(60, len(pool))):
            fh.write(f"https://huggingface.co/spaces/{s['space']},{m},{s['found_in_code']},{s['pinned']},,\n")
except Exception:
    print("SPACES STAGE ERROR (continuing):"); traceback.print_exc()
backup("spaces")
wipe_scratch(); disk_report("after spaces")

# =================================================================================================
# D) inference: RELEASE vs LATEST on the same texts (raw probabilities saved)
# =================================================================================================
def run_inference():
    import numpy as np, torch, transformers
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    try:
        import transformers.modeling_utils as _mu; _mu.auto_conversion = lambda *a, **k: None   # no background conversion thread
    except Exception: pass
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} | transformers {transformers.__version__} | device {dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
    if dev == "cpu" and MODE == "full" and not os.environ.get("FORCE_CPU"):
        print("NO GPU -> skipping inference. Turn on the GPU accelerator and run again (it resumes)."); return

    if os.path.exists(f"{OUT}/texts.json"): T = json.load(open(f"{OUT}/texts.json"))
    else:
        sst = load_dataset("stanfordnlp/sst2", split="validation")
        T = [dict(text=s, lang="en", src="sst2", gold=int(y)) for s, y in zip(sst["sentence"], sst["label"])]
        lid = load_dataset("papluca/language-identification", split="test").shuffle(seed=0)
        per = collections.Counter()
        for r in lid:
            if per[r["labels"]] < 100: T.append(dict(text=r["text"], lang=r["labels"], src="papluca", gold=None)); per[r["labels"]] += 1
        json.dump(T, open(f"{OUT}/texts.json", "w"))
    TEXTS = [t["text"] for t in T]; sst_mask = np.array([t["src"] == "sst2" for t in T]); print("texts:", len(TEXTS))
    HEAD = re.compile(r"(classifier|score|out_proj|logits|cls\.|pooler)")

    from transformers import AutoConfig
    def load_tokenizer(repo, rev):
        """Release tokenizer if loadable; else latest tokenizer (flagged); else a fast BERT tokenizer built from vocab.txt (flagged)."""
        err = None
        for r, fb in ((rev, False), (None, True)):
            try: return AutoTokenizer.from_pretrained(repo, revision=r, trust_remote_code=False), fb
            except Exception as e: err = e
        from transformers import BertTokenizerFast, PreTrainedTokenizerFast
        try: return BertTokenizerFast(vocab_file=hf_hub_download(repo, "vocab.txt", revision=rev)), True
        except Exception: pass
        try:   # tokenizer_config names a class this transformers version lacks (e.g. "TokenizersBackend"): use tokenizer.json directly
            return PreTrainedTokenizerFast(tokenizer_file=hf_hub_download(repo, "tokenizer.json", revision=rev)), True
        except Exception: raise err

    @torch.no_grad()
    def predict(repo, rev, rev_latest):
        tok, tok_fallback = load_tokenizer(repo, rev)
        torch.manual_seed(0); cfg_patched = False
        try:
            mdl, info = AutoModelForSequenceClassification.from_pretrained(repo, revision=rev, trust_remote_code=False,
                                                                           output_loading_info=True)
        except (ValueError, RuntimeError) as e:
            # Old release config.json that does not describe its own weights: (a) no "model_type", or
            # (b) no label count, so the default 2-class head does not fit a 3-class checkpoint.
            # Repair ONLY the missing field from the latest config; weights still come from the release revision.
            msg = str(e)
            if "model_type" not in msg and "size mismatch" not in msg: raise
            raw = json.load(open(hf_hub_download(repo, "config.json", revision=rev)))
            latest = AutoConfig.from_pretrained(repo, revision=rev_latest)
            mtype = raw.pop("model_type", None) or latest.model_type
            if "size mismatch" in msg:
                raw.pop("id2label", None); raw.pop("label2id", None); raw["num_labels"] = latest.num_labels
            config = AutoConfig.for_model(mtype, **raw)
            torch.manual_seed(0)
            mdl, info = AutoModelForSequenceClassification.from_pretrained(repo, revision=rev, config=config, trust_remote_code=False,
                                                                           output_loading_info=True)
            cfg_patched = True
        miss = sorted(info.get("missing_keys", []))
        if tok.pad_token is None:      # batching needs a pad id (decoder-style classifiers); applied identically to R and L
            vocab = tok.get_vocab()
            for cand_tok in ("[PAD]", "<pad>", tok.eos_token, tok.unk_token, tok.sep_token, tok.bos_token):
                if cand_tok in ("[PAD]", "<pad>") and cand_tok not in vocab: continue
                if cand_tok is not None: tok.pad_token = cand_tok; break
        if getattr(mdl.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
            mdl.config.pad_token_id = tok.pad_token_id
        mdl = mdl.to(dev).eval(); out = []
        for s in range(0, len(TEXTS), 32):
            enc = tok(TEXTS[s:s + 32], return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
            out.append(torch.softmax(mdl(**enc).logits.float(), -1).cpu())
        P = torch.cat(out).numpy(); lab = {int(k): str(v) for k, v in mdl.config.id2label.items()}
        del mdl; torch.cuda.empty_cache()
        return P, lab, tok_fallback, miss, cfg_patched

    import signal
    # BaseException (not Exception) so that the many "except Exception" blocks inside transformers / huggingface_hub /
    # our own load_tokenizer fallbacks cannot swallow it. The timer repeats every 30 s after the first expiry, in case
    # something still catches it; the flag makes the handler a no-op once the model is finished.
    class _ModelTimeout(BaseException): pass
    _alarm = {"on": False}
    def _on_alarm(signum, frame):
        if _alarm["on"]: raise _ModelTimeout(f"model exceeded {PER_MODEL_TIMEOUT_S}s (skipped)")
    def _disarm():
        _alarm["on"] = False
        try: signal.setitimer(signal.ITIMER_REAL, 0)
        except ValueError: pass
    RES_PATH = f"{OUT}/flip_results_v5.json"
    RES = json.load(open(RES_PATH)) if os.path.exists(RES_PATH) else {}
    dl = {m["id"]: m["downloads"] for m in MODELS}
    pool = cand if MODE == "full" else list(CL.values())          # smoke: test every smoke model
    queue = sorted([c for c in pool if c["gb"] <= MAX_GB and (c["id"] not in RES or RES[c["id"]].get("note", "").startswith("infer-fail"))],
                   key=lambda c: -dl.get(c["id"], 0))
    print("to evaluate:", len(queue), "(most-downloaded first)")
    t0 = time.time()
    for qi, c in enumerate(queue):
        mid = c["id"]
        if qi % 10 == 0: disk_report(f"{qi}/{len(queue)} | {(time.time() - t0) / 60:.0f} min")
        if free_gb() < 6 or free_gb(".") < 1 or du_gb(SCRATCH) > 8:
            wipe_scratch()
            if free_gb() < 4 or free_gb(".") < 1: print("STOPPING EARLY: low disk. Output saved; run again to resume."); break
        if time.time() - t0 > INFER_BUDGET_S: print("STOPPING EARLY: time budget. Output saved; run again to resume."); break
        try:
            try:
                signal.signal(signal.SIGALRM, _on_alarm); _alarm["on"] = True
                signal.setitimer(signal.ITIMER_REAL, PER_MODEL_TIMEOUT_S, 30)
            except ValueError: _alarm["on"] = False                     # not the main thread: no hard limit (should not happen)
            try:
                PR, lR, fbR, mR, cpR = predict(mid, c["rev_R"], c["rev_L"]); PL, lL, fbL, mL, cpL = predict(mid, c["rev_L"], c["rev_L"])
            finally:
                _disarm()
            row = dict(id=mid, v=7, id2label_R=lR, id2label_L=lL, tok_fallback=fbR or fbL, cfg_patched=cpR or cpL,
                       missing_R=mR[:10], missing_L=mL[:10], random_head=any(HEAD.search(k) for k in mR + mL), note="")
            if PR.shape[1] != PL.shape[1]:
                row["note"] = "label space changed"
                print(f"{mid[:52]:52s} LABEL SPACE CHANGED {PR.shape[1]} -> {PL.shape[1]}")
            else:
                aR, aL = PR.argmax(1), PL.argmax(1)
                nR = np.array([lR[i].lower() for i in aR]); nL = np.array([lL[i].lower() for i in aL])
                row.update(flip_idx=float((aR != aL).mean()), flip_idx_sst=float((aR != aL)[sst_mask].mean()),
                           flip_name=float((nR != nL).mean()), max_prob_shift=float(np.abs(PR - PL).max(1).mean()))
                print(f"{mid[:52]:52s} flip(index)={row['flip_idx']:.1%}  sst={row['flip_idx_sst']:.1%}  name={row['flip_name']:.1%}"
                      + ("  [RANDOM HEAD]" if row["random_head"] else "") + ("  [tok fallback]" if row["tok_fallback"] else "") + ("  [cfg patched]" if row["cfg_patched"] else ""))
            np.savez_compressed(f"{OUT}/preds/{mid.replace('/', '__')}.npz", R=PR.astype(np.float16), L=PL.astype(np.float16))
            RES[mid] = row
        except BaseException as e:                                        # includes the per-model timeout
            _disarm()
            if isinstance(e, (KeyboardInterrupt, SystemExit)): raise
            msg = str(e).replace("\n", " ")
            RES[mid] = dict(id=mid, v=7, note=f"infer-fail {type(e).__name__}: {msg[:300]}")
            print("infer-fail", mid, type(e).__name__, msg[:220])
            import gc; gc.collect()
            try: torch.cuda.empty_cache()
            except Exception: pass
        json.dump(RES, open(RES_PATH, "w"))
        wipe_scratch()
        if qi % 20 == 19: backup(f"inference {qi + 1}/{len(queue)}")

    ok = [r for r in RES.values() if "flip_idx" in r and not r.get("random_head")]
    print(f"\nevaluated {len(ok)} (random-head excluded: {sum(bool(r.get('random_head')) for r in RES.values())}) | "
          f"flip>1%: {sum(r['flip_idx'] > .01 for r in ok)} | flip>10%: {sum(r['flip_idx'] > .10 for r in ok)} | "
          f"name-only changes: {sum(r['flip_idx'] == 0 and r['flip_name'] > 0 for r in ok)} | "
          f"label-space breaks: {sum(r.get('note') == 'label space changed' for r in RES.values())} | "
          f"fails: {sum(r.get('note', '').startswith('infer-fail') for r in RES.values())}")

try:
    run_inference()
except Exception:
    print("INFERENCE STAGE ERROR (output still saved):"); traceback.print_exc()

# =================================================================================================
# F) save
# =================================================================================================
wipe_scratch(); disk_report("end"); save_zip()
backup("final", preds=True)
if BACKUP_REPO:
    try:
        api.upload_file(path_or_fileobj=f"{OUT}_results.zip", path_in_repo=f"{OUT}_results.zip",
                        repo_type="dataset", repo_id=BACKUP_REPO)
    except Exception as e: print("zip backup failed:", type(e).__name__)
    print(f"RESULTS ALSO AT: https://huggingface.co/datasets/{BACKUP_REPO}  (private, Files tab)")
if MODE == "smoke":
    print("\nSMOKE TEST DONE. Paste everything above to Claude. If it looks right, run Save Version > Save & Run All (Commit).")
elif PLATFORM == "colab":
    from google.colab import files; files.download(f"{OUT}_results.zip")
else:
    print(f"DONE. Download {OUT}_results.zip from the Output tab.")
