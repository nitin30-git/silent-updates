# Silent Updates - experiment 2: does the update help or hurt, on the model's OWN task?
# -------------------------------------------------------------------------------------
# Standalone. Kaggle: GPU T4 + Internet ON + secret HF_TOKEN. Runs in roughly 30-60 min.
#
# Cell 1: !pip -q install -U "transformers>=4.46,<5" "huggingface_hub<1.0" datasets safetensors sentencepiece protobuf
# Cell 2: this file.
#
# Experiment 1 showed that predictions change. A reviewer will ask whether they change for the
# better, and whether a flip rate measured on generic sentences means anything for the task the
# model is actually used for. This script answers both: it re-runs the release and latest commits
# of 21 changed models on GOLD-LABELLED data from each model's own task and language, and reports
# accuracy, macro F1 and (for safety classifiers) the share of attacks that slip through.
#
# Label mapping: a model's class names may have been renamed between versions, so we never assume
# an index order. For each version independently we search every mapping from its classes onto the
# gold classes, require the mapping to use every gold class, and keep the one that maximises that
# version's own accuracy. Each version is therefore scored at its best possible orientation, and a
# drop cannot be an artefact of renaming.
#
# Results are printed as JSON at the end as well as saved, so nothing is lost if the output is not.
# -------------------------------------------------------------------------------------
import os, json, time, itertools, traceback, signal

SCRATCH = "/tmp/su2"
os.makedirs(SCRATCH, exist_ok=True)
os.environ.update(
    HF_HUB_CACHE=f"{SCRATCH}/hub", HF_HUB_DISABLE_XET="1", HF_HUB_DISABLE_PROGRESS_BARS="1",
    HF_HUB_DISABLE_TELEMETRY="1", TRANSFORMERS_VERBOSITY="error", TQDM_DISABLE="1",
    TOKENIZERS_PARALLELISM="false", DISABLE_SAFETENSORS_CONVERSION="1", TORCHDYNAMO_DISABLE="1")

PLATFORM = "kaggle" if os.path.exists("/kaggle/working") else "colab" if os.path.exists("/content") else "local"
TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN and PLATFORM == "kaggle":
    from kaggle_secrets import UserSecretsClient; TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
elif not TOKEN and PLATFORM == "colab":
    from google.colab import userdata; TOKEN = userdata.get("HF_TOKEN")
assert TOKEN, "HF_TOKEN not found"
if PLATFORM == "kaggle": os.chdir("/kaggle/working")

import numpy as np, torch, transformers
from huggingface_hub import login
from datasets import load_dataset, get_dataset_config_names
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
login(TOKEN)
try:
    import transformers.modeling_utils as _mu; _mu.auto_conversion = lambda *a, **k: None
except Exception: pass

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"transformers {transformers.__version__} | torch {torch.__version__} | {DEV}")
OUT = "su_gold"; os.makedirs(OUT, exist_ok=True)
MAXLEN, BATCH, PER_MODEL_S, MAX_ROWS = 256, 32, 900, 1500

# ---------------------------------------------------------------- models (revisions from run 1)
MODELS = [
 ("patronus-studio/wolf-defender-prompt-injection", "injection",
  "a8b4d49529cfd466b5d31b75f955912afb8a46b6", "a77e7b669f079c3ad7e72e4ad10f3bd173ac4f4d"),
 ("patronus-studio/wolf-defender-prompt-injection-small", "injection",
  "792be586d1c13c67d1f7b5356546e18ab4bd2400", "bcab2eff97bcabd7227849639e2d0d7a61b46c92"),
 ("dcarpintero/pangolin-guard-base", "injection",
  "e6042ce9ae5c7cbdf72b431fb2788a187af2de5f", "eb220d9f8d75cfbc82cc9d430fa19f85d9764cef"),
 ("NeuralTrust/prompt-guard-oss-small", "injection",
  "bf501c4f765947d9ac3dda1008987fd84ea20b08", "40e5c56b68a1b081484c3e56c9bee726e2749138"),
 ("gbv/mdeberta-ru-prompt-injection", "injection",
  "7f53c98cb222ad226babfd52974a9a6c7b29a579", "546644285dd6c47d40cfcb6a80176dac7ce482c0"),
 ("deepset/deberta-v3-base-injection", "injection",
  "24c047e1af93c63f4c7d75a389a452293e4bf4fb", "80dda00d0b0d9a03917a7685e2ddbcd28e04dbb1"),
 ("eliasalbouzidi/distilbert-nsfw-text-classifier", "toxic_en",
  "af058f9e293f81133ece2f5c0e517d7b016f4222", "c0abb7a10abea1f50528885e9e38d51cca935516"),
 ("cardiffnlp/twitter-roberta-base-hate-latest", "toxic_en",
  "f9839eb15edd2d9851ef50b2ab3e6075c4a94a5b", "cc56585908cbda6d04ba2e1234d911fd1578c9ab"),
 ("unitary/toxic-bert", "toxic_en",
  "d35f11e2888c97eace3a67afee5203fb39b28dea", "4d6c22e74ba2fdd26bc4f7238f50766b045a0d94"),
 ("textdetox/xlmr-large-toxicity-classifier", "toxic_en",
  "eed075bc383528b844017a669300f5e87d0e829a", "b9c7c563427c591fc318d91eb592381ae2fbde66"),
 ("ml6team/distilbert-base-german-cased-toxic-comments", "toxic_de",
  "39b898c15f77c75aa91ce33e6782acbf1004c338", "92d1f1c641db3226d637ab09019a9df44fa007f6"),
 ("MilaNLProc/feel-it-italian-sentiment", "sent_it",
  "a56cda17661eeb6e5bc00cff947a17cc6bbd46fa", "98744f71b7b3a47ba00d57b2736c3af794c417ff"),
 ("UMUTeam/roberta-spanish-sentiment-analysis", "sent_es",
  "cad4207484067458b98b1bda25a71ff0f9e845e4", "fd0be85fc3c8d4812a30730d6f697c33ae8c6f22"),
 ("pysentimiento/robertuito-sentiment-analysis", "sent_es",
  "91676c9328602946c91ed2926a0475033256be08", "a2cc0f67ebd705c55191e25a05ba23d885fcc09b"),
 ("finiteautomata/beto-sentiment-analysis", "sent_es",
  "6204494e41cbc3a434b70fe50b7a260fd048218b", "9384c7f339a6d62c9e2e1d686f225f92e3a3353b"),
 ("finiteautomata/bertweet-base-sentiment-analysis", "sent_en",
  "a1a608261d737b8bc5fa8b1d8feb26d219d3df27", "924fc4c80bccb8003d21fe84dd92c7887717f245"),
 ("tabularisai/multilingual-sentiment-analysis", "sent_es",
  "0b050b7ba02466895b20934de09fd182d543075b", "eea032081f8d247b4303ef3565e7cec1b6f201c9"),
 ("tabularisai/robust-sentiment-analysis", "sent_en",
  "9b9b92e39d74b861dd17ff9532525222090d117d", "c542a281e22b3d840a0b3f6c129acf8e357aed50"),
 ("StephanAkkerman/FinTwitBERT-sentiment", "fin_en",
  "46968276b6afd22937b65f638087dce6761005d8", "da059da3b3bbcb43f9ed1aeb5ae61644010c7e1e"),
 ("beethogedeon/Modern-FinBERT-large", "fin_en",
  "f3da8ccdbed25158944c0eeb6c8f3839774e749f", "a5dd04dae984a46e729698ca39f398181c9aaeea"),
 ("neoyipeng/ModernFinBERT-base", "fin_en",
  "8c62d96150fb5316c78bf64dcc9bb5d59ad7a441", "71404b76e6148b499b89d59d251e64fea52d8760"),
]

# ---------------------------------------------------------------- gold datasets per family
# Each entry is (repo, config, split, language-filter). Families marked POOL concatenate every
# source that loads, because their label semantics are identical (0 = benign/clean, 1 = attack/toxic);
# the sentiment families take the first source that loads, since mixing languages would be wrong.
DATASETS = {
 "injection": [("deepset/prompt-injections", None, "test", None),
               ("jackhhao/jailbreak-classification", None, "test", None),
               ("xTRam1/safe-guard-prompt-injection", None, "test", None)],
 "toxic_en":  [("textdetox/multilingual_toxicity_dataset", None, "en", None),
               ("textdetox/multilingual_toxicity_dataset", None, "train", "en"),
               ("OxAISH-AL-LLM/wiki_toxic", None, "test", None),
               ("google/civil_comments", None, "test", None)],
 "toxic_de":  [("textdetox/multilingual_toxicity_dataset", None, "de", None),
               ("textdetox/multilingual_toxicity_dataset", None, "train", "de")],
 "sent_en":   [("cardiffnlp/tweet_sentiment_multilingual", "english", "test", None),
               ("tyqiangz/multilingual-sentiments", "english", "test", None)],
 "sent_es":   [("cardiffnlp/tweet_sentiment_multilingual", "spanish", "test", None),
               ("tyqiangz/multilingual-sentiments", "spanish", "test", None)],
 "sent_it":   [("cardiffnlp/tweet_sentiment_multilingual", "italian", "test", None),
               ("tyqiangz/multilingual-sentiments", "italian", "test", None)],
 "fin_en":    [("zeroshot/twitter-financial-news-sentiment", None, "validation", None)],
}
POOL = {"injection", "toxic_en", "toxic_de"}
TEXT_COLS = ["text", "sentence", "tweet", "content", "prompt", "comment_text", "question", "body"]
LAB_COLS = ["label", "labels", "toxic", "toxicity", "is_toxic", "type", "sentiment", "target", "class"]
LANG_COLS = ["lang", "language", "locale"]

def _hub_parquet(name, cfg, split):
    """Load a dataset from the Hub's auto-generated parquet branch. Recent `datasets` releases
    refuse repositories that ship a loading script, but every public dataset is mirrored as parquet
    on refs/convert/parquet, which has no script and loads normally."""
    from huggingface_hub import HfApi
    files = [f for f in HfApi().list_repo_files(name, repo_type="dataset",
                                                revision="refs/convert/parquet") if f.endswith(".parquet")]
    if not files: raise FileNotFoundError("no parquet files on refs/convert/parquet")
    sel = [f for f in files if cfg and f.split("/")[0] == cfg] or files
    sel2 = [f for f in sel if f"/{split}/" in f or f"/{split}-" in f or f.split("/")[-1].startswith(split)]
    sel = sel2 or sel
    urls = [f"hf://datasets/{name}@refs/convert/parquet/{f}" for f in sorted(sel)[:16]]
    return load_dataset("parquet", data_files=urls, split="train")

def _one(name, cfg, split, langf, fam):
    """Return (texts, gold, names, source) for a single source, or None."""
    ds = None
    for how in ("direct", "parquet"):
        try:
            if how == "direct":
                ds = load_dataset(name, cfg, split=split) if cfg else load_dataset(name, split=split)
            else:
                ds = _hub_parquet(name, cfg, split)
            break
        except Exception as e:
            print(f"    [{fam}] {name}/{cfg}/{split} via {how} -> {type(e).__name__}: {str(e)[:100]}")
    if ds is None: return None
    cols = ds.column_names
    tc = next((c for c in TEXT_COLS if c in cols), None)
    lc = next((c for c in LAB_COLS if c in cols), None)
    if tc is None or lc is None:
        print(f"    [{fam}] {name}: no usable text/label column in {cols}"); return None
    texts = [str(t) for t in ds[tc]]; raw = list(ds[lc])
    langs = list(ds[next((c for c in LANG_COLS if c in cols), "")]) if any(c in cols for c in LANG_COLS) else None
    if isinstance(raw[0], str):
        vocab = sorted(set(raw)); gold = np.array([vocab.index(v) for v in raw]); names = vocab
    elif isinstance(raw[0], float):
        gold = np.array([int(v >= .5) for v in raw]); names = ["clean", "toxic"]
    else:
        gold = np.array([int(v) for v in raw])
        feat = ds.features[lc]
        names = list(getattr(feat, "names", [])) or [str(v) for v in sorted(set(gold.tolist()))]
    keep = [i for i, t in enumerate(texts) if t and t.strip()]
    if langf and langs: keep = [i for i in keep if str(langs[i]).lower().startswith(langf)]
    if not keep:
        print(f"    [{fam}] {name}: no rows after filtering (lang={langf})"); return None
    print(f"    [{fam}] {name} {cfg or ''} {split}{' lang=' + langf if langf else ''}: "
          f"{len(keep)} rows, classes {names}")
    return ([texts[i] for i in keep], gold[keep], names, f"{name}/{cfg or '-'}/{split}")

def load_gold(fam):
    got = []
    for name, cfg, split, langf in DATASETS[fam]:
        r = _one(name, cfg, split, langf, fam)
        if r: got.append(r)
        if r and fam not in POOL: break
    if not got:
        print(f"    [{fam}] NO DATASET LOADED"); return None
    texts = [t for g in got for t in g[0]]
    gold = np.concatenate([g[1] for g in got])
    names = got[0][2]; source = " + ".join(g[3] for g in got)
    if len(texts) > MAX_ROWS:                      # stratified subsample keeps the class balance
        rng = np.random.default_rng(0); idx = []
        for c in sorted(set(gold.tolist())):
            ci = np.where(gold == c)[0]
            take = max(1, round(MAX_ROWS * len(ci) / len(gold)))
            idx += rng.choice(ci, min(take, len(ci)), replace=False).tolist()
        idx = sorted(idx); texts = [texts[i] for i in idx]; gold = gold[idx]
    uniq = sorted(set(gold.tolist())); remap = {v: i for i, v in enumerate(uniq)}
    gold = np.array([remap[v] for v in gold.tolist()])
    names = [names[v] if v < len(names) else str(v) for v in uniq]
    print(f"    [{fam}] FINAL: {len(texts)} rows, classes {names}, counts {np.bincount(gold).tolist()}")
    return dict(source=source, texts=texts, gold=gold, names=names)

# ---------------------------------------------------------------- scoring
def best_mapping(pred, gold, k, g):
    """Map each of the model's k classes onto one of g gold classes, using every gold class at
    least once, choosing the mapping that maximises accuracy. Returns (accuracy, mapping, mapped)."""
    best = (-1.0, None, None)
    if g ** k > 100000: return best
    for combo in itertools.product(range(g), repeat=k):
        if len(set(combo)) < g: continue                    # must be surjective
        mapped = np.array(combo)[pred]
        acc = float((mapped == gold).mean())
        if acc > best[0]: best = (acc, combo, mapped)
    return best

def macro_f1(y, yh, g):
    fs = []
    for c in range(g):
        tp = int(((yh == c) & (y == c)).sum()); fp = int(((yh == c) & (y != c)).sum())
        fn = int(((yh != c) & (y == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(fs))

def mcnemar(cR, cL):
    """Exact two-sided McNemar on paired correctness vectors."""
    b = int((cR & ~cL).sum()); c = int((~cR & cL).sum())
    n = b + c
    if n == 0: return 1.0, b, c
    from math import comb
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return float(min(1.0, p)), b, c

class TO(BaseException): pass
def _alarm(s, f): raise TO("timeout")

@torch.no_grad()
def predict(repo, rev, texts, rev_latest):
    # Same fallback chain as the main study: the release revision first, then the latest tokenizer,
    # then a tokenizer built directly from the stored vocabulary files. Some repositories name a
    # tokenizer class that this transformers version does not have, which only the last step survives.
    from huggingface_hub import hf_hub_download as _dl
    from transformers import BertTokenizerFast, PreTrainedTokenizerFast
    tok = None
    for r in (rev, None):
        try: tok = AutoTokenizer.from_pretrained(repo, revision=r, trust_remote_code=False); break
        except Exception: pass
    if tok is None:
        for build in (lambda: PreTrainedTokenizerFast(tokenizer_file=_dl(repo, "tokenizer.json", revision=rev)),
                      lambda: BertTokenizerFast(vocab_file=_dl(repo, "vocab.txt", revision=rev))):
            try: tok = build(); break
            except Exception: pass
    if tok is None: raise RuntimeError("tokenizer unavailable")
    torch.manual_seed(0)
    try:
        mdl, info = AutoModelForSequenceClassification.from_pretrained(
            repo, revision=rev, trust_remote_code=False, output_loading_info=True)
    except (ValueError, RuntimeError) as e:
        msg = str(e)
        if "model_type" not in msg and "size mismatch" not in msg: raise
        from huggingface_hub import hf_hub_download
        raw = json.load(open(hf_hub_download(repo, "config.json", revision=rev)))
        latest = AutoConfig.from_pretrained(repo, revision=rev_latest)
        mt = raw.pop("model_type", None) or latest.model_type
        if "size mismatch" in msg:
            raw.pop("id2label", None); raw.pop("label2id", None); raw["num_labels"] = latest.num_labels
        torch.manual_seed(0)
        mdl, info = AutoModelForSequenceClassification.from_pretrained(
            repo, revision=rev, config=AutoConfig.for_model(mt, **raw), trust_remote_code=False,
            output_loading_info=True)
    if tok.pad_token is None:
        v = tok.get_vocab()
        for c in ("[PAD]", "<pad>", tok.eos_token, tok.unk_token, tok.sep_token):
            if c and (c in v or c not in ("[PAD]", "<pad>")): tok.pad_token = c; break
    if getattr(mdl.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
        mdl.config.pad_token_id = tok.pad_token_id
    mdl = mdl.to(DEV).eval()
    out = []
    for s in range(0, len(texts), BATCH):
        enc = tok(texts[s:s + BATCH], return_tensors="pt", padding=True, truncation=True,
                  max_length=MAXLEN).to(DEV)
        out.append(mdl(**enc).logits.float().argmax(-1).cpu())
    pred = torch.cat(out).numpy()
    lab = {int(k): str(v) for k, v in mdl.config.id2label.items()}
    miss = sorted(info.get("missing_keys", []))
    del mdl; torch.cuda.empty_cache()
    return pred, lab, miss

# ---------------------------------------------------------------- run
GOLD = {}
print("loading gold datasets")
for fam in sorted({f for _, f, _, _ in MODELS}):
    GOLD[fam] = load_gold(fam)

RES = json.load(open(f"{OUT}/gold_results.json")) if os.path.exists(f"{OUT}/gold_results.json") else {}
revmap = {m: dict(R=r, L=l) for m, _, r, l in MODELS}
print(f"\n{'model':<52s} {'n':>5s} {'accR':>6s} {'accL':>6s} {'delta':>7s} {'p':>7s} {'disag':>6s}")
for mid, fam, _rev_r, _rev_l in MODELS:
    if mid in RES and "acc_R" in RES[mid]: continue
    G = GOLD.get(fam)
    if G is None: RES[mid] = dict(id=mid, fam=fam, note="no gold dataset"); continue
    info = revmap.get(mid)
    if not info: RES[mid] = dict(id=mid, fam=fam, note="no revisions recorded"); continue
    try:
        signal.signal(signal.SIGALRM, _alarm); signal.setitimer(signal.ITIMER_REAL, PER_MODEL_S, 30)
        try:
            pR, lR, mR = predict(mid, info["R"], G["texts"], info["L"])
            pL, lL, mL = predict(mid, info["L"], G["texts"], info["L"])
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        g = len(set(G["gold"].tolist()))
        aR, mapR, yR = best_mapping(pR, G["gold"], len(lR), g)
        aL, mapL, yL = best_mapping(pL, G["gold"], len(lL), g)
        if mapR is None or mapL is None:
            RES[mid] = dict(id=mid, fam=fam, note="too many classes to map"); continue
        cR, cL = (yR == G["gold"]), (yL == G["gold"])
        p, b, c = mcnemar(cR, cL)
        row = dict(id=mid, fam=fam, source=G["source"], n=len(G["texts"]), gold_classes=G["names"],
                   labels_R=lR, labels_L=lL, map_R=list(mapR), map_L=list(mapL),
                   acc_R=aR, acc_L=aL, f1_R=macro_f1(G["gold"], yR, g), f1_L=macro_f1(G["gold"], yL, g),
                   disagree=float((yR != yL).mean()), mcnemar_p=p, R_only_correct=b, L_only_correct=c,
                   random_head=bool(mR or mL))
        if g == 2:   # safety framing: class 1 is the positive (attack / toxic) class
            pos = 1
            row["miss_rate_R"] = float((yR[G["gold"] == pos] != pos).mean())
            row["miss_rate_L"] = float((yL[G["gold"] == pos] != pos).mean())
            row["fp_rate_R"] = float((yR[G["gold"] != pos] == pos).mean())
            row["fp_rate_L"] = float((yL[G["gold"] != pos] == pos).mean())
        RES[mid] = row
        print(f"{mid[:52]:<52s} {row['n']:5d} {aR:6.1%} {aL:6.1%} {aL-aR:+7.1%} {p:7.4f} {row['disagree']:6.1%}")
    except BaseException as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit)) and not isinstance(e, TO): raise
        RES[mid] = dict(id=mid, fam=fam, note=f"fail {type(e).__name__}: {str(e)[:200]}")
        print(f"{mid[:52]:<52s} FAIL {type(e).__name__}: {str(e)[:90]}")
    json.dump(RES, open(f"{OUT}/gold_results.json", "w"), indent=1)

print("\n==================== SUMMARY ====================")
ev = [r for r in RES.values() if "acc_R" in r]
if ev:
    worse = [r for r in ev if r["acc_L"] < r["acc_R"] - .005]
    better = [r for r in ev if r["acc_L"] > r["acc_R"] + .005]
    sig = [r for r in ev if r["mcnemar_p"] < .05]
    print(f"evaluated {len(ev)} | accuracy down {len(worse)} | up {len(better)} | "
          f"unchanged {len(ev)-len(worse)-len(better)} | significant (p<.05) {len(sig)}")
    print(f"mean accuracy change: {np.mean([r['acc_L']-r['acc_R'] for r in ev]):+.1%}")
    gu = [r for r in ev if r["fam"].startswith(("injection", "toxic")) and "miss_rate_R" in r]
    if gu:
        print(f"\nsafety classifiers: attacks/toxic items missed")
        for r in sorted(gu, key=lambda r: -(r["miss_rate_L"] - r["miss_rate_R"])):
            print(f"  {r['id'][:50]:<50s} miss {r['miss_rate_R']:5.1%} -> {r['miss_rate_L']:5.1%} "
                  f"({r['miss_rate_L']-r['miss_rate_R']:+.1%})   fp {r['fp_rate_R']:5.1%} -> {r['fp_rate_L']:5.1%}")
print("\n---------- RESULTS JSON (copy this if the output tab fails) ----------")
print(json.dumps(RES))
try:
    import shutil; shutil.make_archive("su_gold_results", "zip", ".", OUT); print("\nsaved su_gold_results.zip")
except Exception as e: print("zip failed:", e)
