"""Ship of Theseus between two language models, run on Kaggle (two T4 GPUs).

Ship A is a Qwen 2.5 Instruct model and ship B is the DeepSeek R1 distill of the same size. Their
31 planks (embedding, 28 decoder blocks, final norm, output head) are swapped one at a time and every
hybrid is measured for self-report identity, behavioural fingerprint, perplexity, ARC-Easy accuracy
and free self-reports. See DESIGN.md for the full design. MODE is set by build.sh.
"""
import gc, json, math, os, random, re, shutil, sys, time
from pathlib import Path

MODE = "pilot"  # replaced by build.sh: "pilot" (1.5B pair) or "main" (7B pair)
START = time.time()
BUDGET_H = 11.3
OUT = Path("/kaggle/working/results") if Path("/kaggle").exists() else Path("results")
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "log.txt", "a")


def log(*a):
    s = f"[{(time.time() - START) / 60:7.1f} min] " + " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def time_left_h():
    return BUDGET_H - (time.time() - START) / 3600


PAIRS = {
    "pilot": ("Qwen/Qwen2.5-1.5B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    "main": ("Qwen/Qwen2.5-7B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
}
REPO_A, REPO_B = PAIRS[MODE]
ORDERS = ["forward", "backward", "random1", "random1_rev", "random2", "random2_rev", "random3", "random3_rev"]
if MODE == "pilot":
    ORDERS = ["forward", "backward"]

import torch
import torch.nn.functional as F
from safetensors import safe_open

torch.backends.cuda.matmul.allow_tf32 = True
DT = torch.float16
NGPU = torch.cuda.device_count()
log("mode", MODE, "gpus", NGPU, [torch.cuda.get_device_name(i) for i in range(NGPU)], "torch", torch.__version__)

# ---------------------------------------------------------------- download

def download(repo):
    from huggingface_hub import snapshot_download
    root = Path("/tmp/models") / repo.replace("/", "__")
    p = snapshot_download(repo, local_dir=str(root), allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"])
    return Path(p)


free_gb = shutil.disk_usage("/tmp").free / 1e9
log("free disk on /tmp GB", round(free_gb, 1))
DIR_A, DIR_B = download(REPO_A), download(REPO_B)
CFG_A = json.loads((DIR_A / "config.json").read_text())
CFG_B = json.loads((DIR_B / "config.json").read_text())
for k in ["hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "intermediate_size", "vocab_size"]:
    assert CFG_A[k] == CFG_B[k], k
THETA = {"A": float(CFG_A["rope_theta"]), "B": float(CFG_B["rope_theta"])}
EPS = {"A": float(CFG_A["rms_norm_eps"]), "B": float(CFG_B["rms_norm_eps"])}
H, NL = CFG_A["hidden_size"], CFG_A["num_hidden_layers"]
NH, NKV = CFG_A["num_attention_heads"], CFG_A["num_key_value_heads"]
HD = H // NH
log("theta", THETA, "eps", EPS, "layers", NL)

from transformers import AutoTokenizer
TOK = AutoTokenizer.from_pretrained(str(DIR_A))


def enc(s):
    return TOK.encode(s, add_special_tokens=False)


# Special ids shared by both vocabularies with different meanings.
EOT, IM_START, IM_END = 151643, 151644, 151645  # A: <|endoftext|>, <|im_start|>, <|im_end|>
B_BOS, B_USER, B_ASSIST, B_THINK, B_ENDTHINK = 151646, 151644, 151645, 151648, 151649
STOP = {151643, 151645}


def tmpl_A(user):
    return ([IM_START] + enc("system\nYou are a helpful assistant.") + [IM_END] + enc("\n") + [IM_START]
            + enc("user\n" + user) + [IM_END] + enc("\n") + [IM_START] + enc("assistant\n"))


def tmpl_B(user):
    return [B_BOS, B_USER] + enc(user) + [B_ASSIST, B_THINK] + enc("\n")


def tmpl_B_nothink(user):
    return tmpl_B(user) + enc("\n") + [B_ENDTHINK] + enc("\n\n")


TEMPLATES = {"A": tmpl_A, "B": tmpl_B}

# ---------------------------------------------------------------- weights


class Source:
    def __init__(self, d):
        idx = d / "model.safetensors.index.json"
        if idx.exists():
            wm = json.loads(idx.read_text())["weight_map"]
        else:
            with safe_open(str(d / "model.safetensors"), "pt") as f:
                wm = {k: "model.safetensors" for k in f.keys()}
        self.wm = wm
        self.handles = {fn: safe_open(str(d / fn), "pt", device="cpu") for fn in set(wm.values())}

    def get(self, name):
        if name == "lm_head.weight" and name not in self.wm:
            name = "model.embed_tokens.weight"  # tied head
        return self.handles[self.wm[name]].get_tensor(name)


SRC = {"A": Source(DIR_A), "B": Source(DIR_B)}

LAYER_KEYS = {
    "q_w": "self_attn.q_proj.weight", "q_b": "self_attn.q_proj.bias",
    "k_w": "self_attn.k_proj.weight", "k_b": "self_attn.k_proj.bias",
    "v_w": "self_attn.v_proj.weight", "v_b": "self_attn.v_proj.bias",
    "o_w": "self_attn.o_proj.weight",
    "ln1": "input_layernorm.weight", "ln2": "post_attention_layernorm.weight",
    "gate": "mlp.gate_proj.weight", "up": "mlp.up_proj.weight", "down": "mlp.down_proj.weight",
}
ATTN = ["q_w", "q_b", "k_w", "k_b", "v_w", "v_b", "o_w"]
MLP = ["gate", "up", "down"]
NORMS = ["ln1", "ln2"]


def hf_name(key):
    if key == "embed":
        return "model.embed_tokens.weight"
    if key == "norm":
        return "model.norm.weight"
    if key == "head":
        return "lm_head.weight"
    _, i, k = key.split(".")
    return f"model.layers.{i}.{LAYER_KEYS[k]}"


ALL_KEYS = ["embed"] + [f"L.{i}.{k}" for i in range(NL) for k in LAYER_KEYS] + ["norm", "head"]
PLANKS = [["embed"]] + [[f"L.{i}.{k}" for k in LAYER_KEYS] for i in range(NL)] + [["norm"], ["head"]]
NP = len(PLANKS)  # 31


def device_of(key):
    if NGPU <= 1:
        return torch.device("cuda:0")
    if key == "embed":
        return torch.device("cuda:0")
    if key in ("norm", "head"):
        return torch.device(f"cuda:{NGPU - 1}")
    i = int(key.split(".")[1])
    return torch.device(f"cuda:{min(i * NGPU // NL, NGPU - 1)}")


def layer_dev(i):
    return device_of(f"L.{i}.q_w")


def spec_theta(spec):
    if spec == "A":
        return THETA["A"]
    if spec == "B":
        return THETA["B"]
    a = spec[1]
    return 10 ** ((1 - a) * math.log10(THETA["A"]) + a * math.log10(THETA["B"]))


class Ship:
    """Plain Qwen2 forward pass whose every tensor can come from A, B or a mix."""

    def __init__(self):
        self.w, self.spec = {}, {}
        for key in ALL_KEYS:
            t = SRC["A"].get(hf_name(key)).to(DT)
            self.w[key] = t.to(device_of(key))
            self.spec[key] = "A"
        self.theta = [THETA["A"]] * NL
        self.theta_override = None
        self.rope_cache = {}

    def set_tensor(self, key, spec):
        if self.spec[key] == spec:
            return
        if spec in ("A", "B"):
            t = SRC[spec].get(hf_name(key))
        else:
            a = spec[1]
            t = (1 - a) * SRC["A"].get(hf_name(key)).float() + a * SRC["B"].get(hf_name(key)).float()
        self.w[key].copy_(t.to(DT))
        self.spec[key] = spec

    def set_state(self, state, theta_override=None):
        """state: dict key -> 'A' | 'B' | ('mix', a). Rotary base follows the q projection."""
        for key in ALL_KEYS:
            self.set_tensor(key, state[key])
        self.theta = [spec_theta(state[f"L.{i}.q_w"]) for i in range(NL)]
        self.theta_override = theta_override

    def rope(self, theta, pos, dev):
        inv = 1.0 / (theta ** (torch.arange(0, HD, 2, device=dev, dtype=torch.float32) / HD))
        fr = pos.to(dev).float()[..., None] * inv
        emb = torch.cat([fr, fr], -1)
        return emb.cos().to(DT)[:, None], emb.sin().to(DT)[:, None]

    @staticmethod
    def rot(x, cos, sin):
        x1, x2 = x[..., : HD // 2], x[..., HD // 2:]
        return x * cos + torch.cat([-x2, x1], -1) * sin

    def rms(self, x, w):
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS["A"])
        return w * x.to(DT)

    @torch.no_grad()
    def run(self, ids, key_valid, pos, cache=None):
        """ids [B,T]; key_valid [B,S] for all keys incl. past; pos [B,T]. Returns normed hidden [B,T,H]."""
        Bsz, T = ids.shape
        S = key_valid.shape[1]
        past = S - T
        h = F.embedding(ids.to(self.w["embed"].device), self.w["embed"]).float()
        masks, ropes = {}, {}
        for i in range(NL):
            dev = layer_dev(i)
            if h.device != dev:
                h = h.to(dev)
            if dev not in masks:
                kv = key_valid.to(dev)
                qi = torch.arange(T, device=dev)[:, None] + past
                ki = torch.arange(S, device=dev)[None, :]
                m = (ki <= qi)[None, None] & kv[:, None, None, :]
                m = m | (ki == qi)[None, None]
                masks[dev] = m
            th = self.theta_override if self.theta_override is not None else self.theta[i]
            if (th, dev) not in ropes:
                ropes[(th, dev)] = self.rope(th, pos, dev)
            cos, sin = ropes[(th, dev)]
            p = f"L.{i}."
            w = self.w
            x = self.rms(h, w[p + "ln1"])
            q = F.linear(x, w[p + "q_w"], w[p + "q_b"]).view(Bsz, T, NH, HD).transpose(1, 2)
            k = F.linear(x, w[p + "k_w"], w[p + "k_b"]).view(Bsz, T, NKV, HD).transpose(1, 2)
            v = F.linear(x, w[p + "v_w"], w[p + "v_b"]).view(Bsz, T, NKV, HD).transpose(1, 2)
            q, k = self.rot(q, cos, sin), self.rot(k, cos, sin)
            if cache is not None:
                if cache[i] is not None:
                    k = torch.cat([cache[i][0], k], 2)
                    v = torch.cat([cache[i][1], v], 2)
                cache[i] = (k, v)
            kr = k.repeat_interleave(NH // NKV, 1)
            vr = v.repeat_interleave(NH // NKV, 1)
            a = F.scaled_dot_product_attention(q, kr, vr, attn_mask=masks[dev])
            a = a.transpose(1, 2).reshape(Bsz, T, H)
            h = h + F.linear(a, w[p + "o_w"]).float()
            x = self.rms(h, w[p + "ln2"])
            h = h + F.linear(F.silu(F.linear(x, w[p + "gate"])) * F.linear(x, w[p + "up"]), w[p + "down"]).float()
        h = h.to(self.w["norm"].device)
        return self.rms(h, self.w["norm"])

    def logits(self, hsel):
        return F.linear(hsel, self.w["head"]).float()


# ---------------------------------------------------------------- batched helpers

NONFINITE = [0]


def batches(n_items, lens, budget):
    order = sorted(range(n_items), key=lambda i: -lens[i])
    cur, cur_max = [], 0
    for i in order:
        m = max(cur_max, lens[i])
        if cur and m * (len(cur) + 1) > budget:
            yield cur
            cur, m = [], lens[i]
        cur.append(i)
        cur_max = m
    if cur:
        yield cur


TOKEN_BUDGET = 6144 if MODE == "main" else 12288


def pad_left(seqs):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), EOT, dtype=torch.long)
    valid = torch.zeros((len(seqs), L), dtype=torch.bool)
    for r, s in enumerate(seqs):
        ids[r, L - len(s):] = torch.tensor(s)
        valid[r, L - len(s):] = True
    pos = (valid.long().cumsum(1) - 1).clamp(min=0)
    return ids, valid, pos


@torch.no_grad()
def token_logprobs(ship, seqs, starts):
    """For each seq, logprobs of tokens seq[start:], each predicted from its prefix."""
    out = [None] * len(seqs)
    for b in batches(len(seqs), [len(s) for s in seqs], TOKEN_BUDGET):
        ids, valid, pos = pad_left([seqs[i] for i in b])
        hN = ship.run(ids.cuda(), valid, pos)
        L = ids.shape[1]
        for r, i in enumerate(b):
            s = seqs[i]
            off = L - len(s)
            p0, p1 = off + starts[i] - 1, off + len(s) - 1
            lg = ship.logits(hN[r, p0:p1])
            NONFINITE[0] += int((~torch.isfinite(lg)).sum())
            lp = torch.log_softmax(lg, -1)
            tgt = torch.tensor(s[starts[i]:], device=lp.device)
            out[i] = lp.gather(1, tgt[:, None])[:, 0].cpu().tolist()
        del hN
    return out


@torch.no_grad()
def last_logprobs(ship, seqs, cands):
    out = [None] * len(seqs)
    for b in batches(len(seqs), [len(s) for s in seqs], TOKEN_BUDGET):
        ids, valid, pos = pad_left([seqs[i] for i in b])
        hN = ship.run(ids.cuda(), valid, pos)
        lg = ship.logits(hN[:, -1])
        NONFINITE[0] += int((~torch.isfinite(lg)).sum())
        lp = torch.log_softmax(lg, -1)
        for r, i in enumerate(b):
            out[i] = [float(lp[r, c]) for c in cands[i]]
    return out


@torch.no_grad()
def generate(ship, prompts, max_new):
    ids, valid, pos = pad_left(prompts)
    Bsz = len(prompts)
    cache = [None] * NL
    hN = ship.run(ids.cuda(), valid, pos, cache)
    nxt = ship.logits(hN[:, -1]).argmax(-1).cpu()
    outs = [[] for _ in range(Bsz)]
    done = [False] * Bsz
    key_valid, last_pos = valid, pos[:, -1]
    for step in range(max_new):
        for r in range(Bsz):
            if not done[r]:
                outs[r].append(int(nxt[r]))
                if int(nxt[r]) in STOP:
                    done[r] = True
        if all(done) or step == max_new - 1:
            break
        key_valid = torch.cat([key_valid, torch.ones(Bsz, 1, dtype=torch.bool)], 1)
        last_pos = last_pos + 1
        hN = ship.run(nxt[:, None].cuda(), key_valid, last_pos[:, None], cache)
        lg = ship.logits(hN[:, -1])
        NONFINITE[0] += int((~torch.isfinite(lg)).sum())
        nxt = lg.argmax(-1).cpu()
    del cache
    return outs


SPECIAL_NAMES = {151643: "<eot>", 151644: "<im_start|User>", 151645: "<im_end|Assistant>", 151646: "<B_bos>",
                 151647: "<EOT>", 151648: "<think>", 151649: "</think>"}


def decode(ids):
    parts, buf = [], []
    for t in ids:
        if t in SPECIAL_NAMES or t >= 151643:
            if buf:
                parts.append(TOK.decode(buf))
                buf = []
            parts.append(SPECIAL_NAMES.get(t, f"<{t}>"))
        else:
            buf.append(t)
    if buf:
        parts.append(TOK.decode(buf))
    return "".join(parts)


# ---------------------------------------------------------------- probes and data

NAME_Q = ["What is your name?", "What should I call you?", "Tell me your name.", "Which AI model are you?",
          "What model am I talking to right now?", "Please state your name.", "What are you called?",
          "Who are you, by name?"]
MAKER_Q = ["Who created you?", "Which company made you?", "Who trained you?", "Who built you?",
           "Which organization developed you?", "Who is your developer?",
           "Where do you come from? Which lab made you?", "Who designed you?"]
NAME_C = [" Qwen", " DeepSeek", " ChatGPT", " Claude", " Llama", " Gemini"]
MAKER_C = [" Alibaba", " DeepSeek", " OpenAI", " Anthropic", " Meta", " Google"]
OPEN_Q = ["Who are you?", "Introduce yourself in two sentences.", "What is your name, and who made you?",
          "Are you ChatGPT?", "Describe what kind of AI you are.", "What model are you, exactly?",
          "Tell me something about your origins.", "If you had to sign a letter, what name would you sign?"]
FP_PROMPTS = [
    "Explain how photosynthesis works.", "What causes the seasons on Earth?", "Give me three tips for sleeping better.",
    "Write a haiku about rain.", "What is 17 times 23?", "If a train travels 120 km in 1.5 hours, what is its average speed?",
    "Summarise the plot of Romeo and Juliet.", "How do vaccines work?", "What is the difference between weather and climate?",
    "Suggest a name for a small bakery.", "Explain recursion to a beginner programmer.", "Why is the sky blue?",
    "What is the capital of Australia, and why is it not Sydney?", "Write a short apology email for missing a meeting.",
    "How many prime numbers are there between 1 and 30?", "Explain what inflation is.", "Give me a recipe for a simple omelette.",
    "What are the main causes of the French Revolution?", "Translate 'good morning, how are you?' into French.",
    "Solve for x: 3x + 7 = 22.", "What is a black hole?", "How should I prepare for a job interview?",
    "Write two sentences describing autumn.", "What does DNA stand for and what does it do?", "Explain the rules of chess briefly.",
    "Is it better to rent or buy a house?", "What is the Pythagorean theorem?", "Describe the water cycle.",
    "Give me a motivational quote and explain it.", "How does a refrigerator keep food cold?", "What is machine learning?",
    "A shop sells pens at 3 for 2 dollars. How much do 12 pens cost?", "What is the tallest mountain in the world?",
    "Write a limerick about a cat.", "How can I reduce stress at work?", "Explain supply and demand.",
    "What is the boiling point of water at sea level in Fahrenheit?", "Why do leaves change colour in autumn?",
    "Describe a good morning routine.", "What were the causes of World War I?", "How do airplanes stay in the air?",
    "What is 2 to the power of 10?", "Write a one-paragraph story about a lost key.", "What is the role of the heart?",
    "Explain what a prime number is.", "How do I make a good first impression?", "What is the speed of light?",
    "Name three renewable energy sources and describe one.",
]
assert len(FP_PROMPTS) == 48


def identity_items():
    items = []
    for t in ("A", "B"):
        for kind, qs, prefix, cands in (("name", NAME_Q, "My name is", NAME_C), ("maker", MAKER_Q, "I was created by", MAKER_C)):
            for q in qs:
                base = (tmpl_A(q) if t == "A" else tmpl_B_nothink(q)) + enc(prefix)
                items.append({"tmpl": t, "kind": kind, "q": q, "base": base, "cands": [enc(c) for c in cands]})
    return items


ID_ITEMS = identity_items()


def load_eval_data():
    from datasets import load_dataset
    wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n".join(x for x in wt["text"] if x.strip())
    ids = enc(text)
    rng = random.Random(0)
    starts = sorted(rng.sample(range(0, len(ids) - 300), 32))
    wiki = [ids[s:s + 256] for s in starts]
    arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    idx = random.Random(0).sample(range(len(arc)), 150)
    arc_items = []
    for i in idx:
        ex = arc[i]
        letters = "ABCDE"[: len(ex["choices"]["text"])]
        prompt = "Question: " + ex["question"] + "\n" + "".join(f"{l}. {c}\n" for l, c in zip(letters, ex["choices"]["text"])) + "Answer:"
        gold = ex["choices"]["label"].index(ex["answerKey"])
        arc_items.append({"ids": enc(prompt), "cands": [enc(" " + l)[0] for l in letters], "gold": gold})
    return wiki, arc_items


# ---------------------------------------------------------------- evaluation of one hybrid

A_RE = re.compile(r"qwen|alibaba|tongyi|通义|千问|阿里", re.I)
B_RE = re.compile(r"deepseek|深度求索", re.I)


def label_text(s):
    a, b = bool(A_RE.search(s)), bool(B_RE.search(s))
    return "both" if a and b else "A" if a else "B" if b else "neither"


def degenerate(ids):
    if len(ids) < 64:
        return False
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return len(set(grams)) / len(grams) < 0.3


def evaluate(ship, refs, wiki, arc_items, do_gen=True):
    t0 = time.time()
    NONFINITE[0] = 0
    r = {}
    # 1. identity forced choice
    seqs, starts, owner = [], [], []
    for j, it in enumerate(ID_ITEMS):
        for c in it["cands"]:
            seqs.append(it["base"] + c)
            starts.append(len(it["base"]))
            owner.append(j)
    lps = token_logprobs(ship, seqs, starts)
    cand_lp = [[] for _ in ID_ITEMS]
    for j, lp in zip(owner, lps):
        cand_lp[j].append(sum(lp))
    r["id_items"] = [{"tmpl": it["tmpl"], "kind": it["kind"], "cand_lp": [round(x, 4) for x in cl],
                      "score": round(cl[0] - cl[1], 4)} for it, cl in zip(ID_ITEMS, cand_lp)]
    for t in ("A", "B"):
        r[f"I_raw_{t}"] = sum(x["score"] for x in r["id_items"] if x["tmpl"] == t) / 16
    # 2. fingerprint
    fp = {}
    for owner_name in ("A", "B"):
        seqs = [p + resp for p, resp in zip(refs[owner_name]["prompts"], refs[owner_name]["responses"])]
        starts = [len(p) for p in refs[owner_name]["prompts"]]
        lps = token_logprobs(ship, seqs, starts)
        fp[owner_name] = [sum(x) / len(x) for x in lps]
    r["fp_items"] = {k: [round(x, 4) for x in v] for k, v in fp.items()}
    r["F_raw"] = sum(fp["A"]) / len(fp["A"]) - sum(fp["B"]) / len(fp["B"])
    # 3. seaworthiness
    lps = token_logprobs(ship, wiki, [1] * len(wiki))
    nll = -sum(sum(x) for x in lps) / sum(len(x) for x in lps)
    r["wiki_ppl"] = math.exp(min(nll, 50))
    al = last_logprobs(ship, [a["ids"] for a in arc_items], [a["cands"] for a in arc_items])
    r["arc_acc"] = sum(int(max(range(len(x)), key=lambda i: x[i]) == a["gold"]) for x, a in zip(al, arc_items)) / len(arc_items)
    # 4. free self-reports
    if do_gen:
        prompts, meta = [], []
        for t in ("A", "B"):
            for q in OPEN_Q:
                prompts.append(TEMPLATES[t](q))
                meta.append((t, q))
        outs = generate(ship, prompts, 256)
        r["gen"] = []
        for (t, q), o in zip(meta, outs):
            txt = decode(o)
            r["gen"].append({"tmpl": t, "q": q, "text": txt, "label": label_text(txt), "degenerate": degenerate(o),
                             "think_tokens": int(sum(1 for x in o if x in (B_THINK, B_ENDTHINK))), "n": len(o)})
    r["nonfinite"] = NONFINITE[0]
    r["secs"] = round(time.time() - t0, 1)
    return r


# ---------------------------------------------------------------- states

def state_all(spec):
    return {k: spec for k in ALL_KEYS}


def state_mask(mask):
    st = {}
    for p, src in zip(PLANKS, mask):
        for k in p:
            st[k] = src
    return st


def order_perm(name):
    if name == "forward":
        return list(range(NP))
    if name == "backward":
        return list(range(NP))[::-1]
    seed = int(name[6])
    perm = list(range(NP))
    random.Random(seed).shuffle(perm)
    return perm[::-1] if name.endswith("_rev") else perm


def kind_state(kind, into):
    """Swap every part of one kind. into='A' means B's parts go into A."""
    base, other = ("A", "B") if into == "A" else ("B", "A")
    st = state_all(base)
    for k in ALL_KEYS:
        sub = k.split(".")[-1]
        if (kind == "attn" and sub in ATTN) or (kind == "mlp" and sub in MLP) or \
           (kind == "norms" and (sub in NORMS or k == "norm")) or (kind == "embhead" and k in ("embed", "head")):
            st[k] = other
    return st


def hybrid_list():
    hs = []
    for o in ORDERS:
        perm = order_perm(o)
        mask = ["A"] * NP
        for k in range(1, NP):
            mask[perm[k - 1]] = "B"
            hs.append({"id": f"arm1/{o}/k{k:02d}", "arm": 1, "order": o, "k": k, "mask": "".join(mask),
                       "state": state_mask(list(mask))})
    for kind in ("attn", "mlp", "norms", "embhead"):
        for into in ("A", "B"):
            hs.append({"id": f"arm2/{kind}_into_{into}", "arm": 2, "kind": kind, "into": into, "state": kind_state(kind, into)})
    for j in range(1, 10):
        a = round(j / 10, 1)
        hs.append({"id": f"arm3/a{a}", "arm": 3, "a": a, "state": state_all(("mix", a))})
    return hs


# ---------------------------------------------------------------- gate 1: compare with Hugging Face


GATE_TEXTS = ["The history of the Roman Empire begins with", "def fibonacci(n):\n    "]


@torch.no_grad()
def hf_reference(d):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(str(d), torch_dtype=DT, device_map="auto")
    m.eval()
    out = []
    seqs = [enc(t) for t in GATE_TEXTS] + [tmpl_A("Explain gravity in one paragraph."), tmpl_B("What is 12 squared?")]
    for s in seqs:
        lg = m(torch.tensor([s]).to(m.device)).logits[0].float().cpu()
        out.append((s, lg))
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def compare(ship, ref):
    agree, tot, dagree, dtot, maxd, finite = 0, 0, 0, 0, 0.0, True
    for s, lg in ref:
        ids, valid, pos = pad_left([s])
        mine = ship.logits(ship.run(ids.cuda(), valid, pos)[0]).cpu()
        finite &= bool(torch.isfinite(mine).all() and torch.isfinite(lg).all())
        same = mine.argmax(-1) == lg.argmax(-1)
        top2 = lg.topk(2, -1).values
        decisive = (top2[:, 0] - top2[:, 1]) > 1.0  # amendment 1: margin larger than the logit tolerance
        agree += int(same.sum())
        tot += len(s)
        dagree += int((same & decisive).sum())
        dtot += int(decisive.sum())
        maxd = max(maxd, float((mine - lg).abs().max()))
    return {"argmax_agree": agree / tot, "decisive_agree": dagree / max(dtot, 1), "n_decisive": dtot,
            "max_abs_logit_diff": maxd, "finite": finite}


# ---------------------------------------------------------------- main


def find_previous(name):
    for p in Path("/kaggle/input").rglob(name) if Path("/kaggle/input").exists() else []:
        return p
    return None


def main():
    res_path = OUT / "hybrids.jsonl"
    prev = find_previous("hybrids.jsonl")
    if prev and not res_path.exists():
        shutil.copy(prev, res_path)
        for nm in ("refs.json", "gates.json"):
            pp = find_previous(nm)
            if pp:
                shutil.copy(pp, OUT / nm)
        log("resumed from", prev)
    done = set()
    if res_path.exists():
        for line in res_path.open():
            done.add(json.loads(line)["id"])
    log("already done", len(done))

    gates_path = OUT / "gates.json"
    if gates_path.exists():
        gates = json.loads(gates_path.read_text())
    else:
        refA, refB = hf_reference(DIR_A), hf_reference(DIR_B)
        ship = Ship()
        g1A = compare(ship, refA)
        ship.set_state(state_all("B"))
        g1B = compare(ship, refB)
        gates = {"g1_A": g1A, "g1_B": g1B}
        del ship, refA, refB
        gc.collect()
        torch.cuda.empty_cache()
        gates["g1_pass"] = all(g["decisive_agree"] >= 0.99 and g["max_abs_logit_diff"] < 1.0 for g in (g1A, g1B))
        gates["g4_pass"] = g1A["finite"] and g1B["finite"]
        gates_path.write_text(json.dumps(gates, indent=1))
    log("gate 1 and 4", gates)

    ship = Ship()
    wiki, arc_items = load_eval_data()
    refs_path = OUT / "refs.json"
    if refs_path.exists():
        refs = json.loads(refs_path.read_text())
    else:
        refs = {}
        for owner in ("A", "B"):
            ship.set_state(state_all(owner))
            prompts = [TEMPLATES[owner](q) for q in FP_PROMPTS]
            outs = []
            for i in range(0, len(prompts), 16):
                outs += generate(ship, prompts[i:i + 16], 192)
            refs[owner] = {"prompts": prompts, "responses": outs, "texts": [decode(o) for o in outs]}
            log("reference answers from", owner, "done; sample:", refs[owner]["texts"][0][:200].replace("\n", " "))
        refs_path.write_text(json.dumps(refs))

    def record(h, r):
        row = {k: v for k, v in h.items() if k != "state"}
        row.update(r)
        with res_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        done.add(h["id"])

    controls = [
        {"id": "ctrl/A", "arm": 0, "state": state_all("A")},
        {"id": "ctrl/B", "arm": 0, "state": state_all("B")},
        {"id": "ctrl/A_thetaB", "arm": 0, "state": state_all("A"), "theta": THETA["B"]},
        {"id": "ctrl/B_thetaA", "arm": 0, "state": state_all("B"), "theta": THETA["A"]},
    ]
    for h in controls:
        if h["id"] in done:
            continue
        ship.set_state(h["state"], h.get("theta"))
        r = evaluate(ship, refs, wiki, arc_items)
        record(h, r)
        log(h["id"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in ("I_raw_A", "I_raw_B", "F_raw", "wiki_ppl", "arc_acc", "nonfinite", "secs")})

    rows = {json.loads(l)["id"]: json.loads(l) for l in res_path.open()}
    cA, cB = rows["ctrl/A"], rows["ctrl/B"]
    gates["g2_sep_A_tmpl"] = cA["I_raw_A"] - cB["I_raw_A"]
    gates["g2_sep_B_tmpl"] = cA["I_raw_B"] - cB["I_raw_B"]
    gates["g2_pass"] = gates["g2_sep_A_tmpl"] >= 4 and gates["g2_sep_B_tmpl"] >= 4
    gates["g3_sep"] = cA["F_raw"] - cB["F_raw"]
    gates["g3_pass"] = gates["g3_sep"] >= 0.5
    gates_path.write_text(json.dumps(gates, indent=1))
    log("gates", gates)
    if MODE == "main" and not all(gates[g] for g in ("g1_pass", "g2_pass", "g3_pass", "g4_pass")):
        log("A GATE FAILED. Stopping before any hybrid is run, as the design requires.")
        return

    ship.set_state(state_all("A"))
    for h in hybrid_list():
        if h["id"] in done:
            continue
        if time_left_h() < 0.25:
            log("time budget reached, stopping; commit again with this version as input to resume")
            break
        ship.set_state(h["state"])
        r = evaluate(ship, refs, wiki, arc_items)
        record(h, r)
        log(h["id"], "I_A %.2f I_B %.2f F %.3f ppl %.1f arc %.2f nf %d %ss" % (
            r["I_raw_A"], r["I_raw_B"], r["F_raw"], r["wiki_ppl"], r["arc_acc"], r["nonfinite"], r["secs"]))
    total = len(hybrid_list()) + len(controls)
    log("finished", len(done), "of", total)
    if len(done) >= total:
        analyze(res_path)


# ---------------------------------------------------------------- analysis


def analyze(res_path):
    import numpy as np
    from scipy.optimize import curve_fit
    rows = [json.loads(l) for l in res_path.open()]
    by = {r["id"]: r for r in rows}
    cA, cB = by["ctrl/A"], by["ctrl/B"]
    rng = np.random.default_rng(0)
    items_A = lambda r, t: np.array([x["score"] for x in r["id_items"] if x["tmpl"] == t])

    def iota(r, t, idx=None):
        a, b, x = items_A(cA, t), items_A(cB, t), items_A(r, t)
        if idx is not None:
            a, b, x = a[idx], b[idx], x[idx]
        return float((a.mean() - x.mean()) / (a.mean() - b.mean()))

    def phi(r):
        return (cA["F_raw"] - r["F_raw"]) / (cA["F_raw"] - cB["F_raw"])

    worse_ppl = max(cA["wiki_ppl"], cB["wiki_ppl"])
    summary = {"parents": {"A": {k: cA[k] for k in ("wiki_ppl", "arc_acc", "I_raw_A", "I_raw_B", "F_raw")},
                           "B": {k: cB[k] for k in ("wiki_ppl", "arc_acc", "I_raw_A", "I_raw_B", "F_raw")}}}
    table = []
    for r in rows:
        table.append({"id": r["id"], "arm": r["arm"], "k": r.get("k"), "order": r.get("order"), "mask": r.get("mask"),
                      "iota_A": iota(r, "A"), "iota_B": iota(r, "B"), "phi": phi(r), "ppl": r["wiki_ppl"],
                      "arc": r["arc_acc"], "seaworthy": r["wiki_ppl"] <= 2 * worse_ppl,
                      "gen_labels": {t: [g["label"] for g in r.get("gen", []) if g["tmpl"] == t] for t in "AB"}})
    tb = {t["id"]: t for t in table}

    def logistic(k, k50, w):
        return 1 / (1 + np.exp(-(k - k50) / w))

    def curve(order, t, idx=None):
        ks = np.arange(0, NP + 1)
        ys = [0.0 if idx is None else iota(cA, t, idx)] + \
             [iota(by[f"arm1/{order}/k{k:02d}"], t, idx) for k in range(1, NP)] + \
             [1.0 if idx is None else iota(cB, t, idx)]
        return ks, np.array(ys)

    def fit(ks, ys):
        try:
            p, _ = curve_fit(logistic, ks, ys, p0=[NP / 2, 2.0], bounds=([-10, 0.05], [NP + 10, 50]), maxfev=5000)
        except Exception:
            return None
        rss_log = float(((ys - logistic(ks, *p)) ** 2).sum())
        A_ = np.vstack([ks, np.ones_like(ks)]).T
        coef, *_ = np.linalg.lstsq(A_, ys, rcond=None)
        rss_lin = float(((ys - A_ @ coef) ** 2).sum())
        n = len(ks)
        return {"k50": float(p[0]), "width": float(p[1]), "dBIC_lin_minus_log": n * math.log(max(rss_lin, 1e-12) / max(rss_log, 1e-12))}

    h1 = {}
    boots = {}
    for t in "AB":
        for o in ("forward", "backward"):
            if f"arm1/{o}/k01" not in by:
                continue
            f0 = fit(*curve(o, t))
            bs = []
            for _ in range(2000 if MODE == "main" else 200):
                idx = rng.integers(0, 16, 16)
                f = fit(*curve(o, t, idx))
                if f:
                    bs.append(f["k50"])
            boots[(t, o)] = np.array(bs)
            if f0:
                f0["k50_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            h1[f"{t}_{o}"] = f0
        if (t, "forward") in boots and (t, "backward") in boots:
            n = min(len(boots[(t, "forward")]), len(boots[(t, "backward")]))
            d = boots[(t, "forward")][:n] - boots[(t, "backward")][:n]
            h1[f"{t}_k50_fwd_minus_bwd_ci"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    summary["H1_H2_curves"] = h1

    arm1 = [t for t in table if t["arm"] == 1]
    X = np.array([[1.0 if c == "B" else 0.0 for c in t["mask"]] for t in arm1])
    infl = {}
    for name, y in (("iota_A", [t["iota_A"] for t in arm1]), ("iota_B", [t["iota_B"] for t in arm1]),
                    ("phi", [t["phi"] for t in arm1]), ("log_ppl", [math.log(t["ppl"]) for t in arm1])):
        y = np.array(y)
        Xc, yc = X - X.mean(0), y - y.mean()
        beta = np.linalg.solve(Xc.T @ Xc + 1.0 * np.eye(NP), Xc.T @ yc)
        pred = Xc @ beta
        infl[name] = {"coef": [round(float(b), 4) for b in beta], "R2": float(1 - ((yc - pred) ** 2).sum() / (yc ** 2).sum())}
    summary["H2_influence"] = infl

    masks = {t["mask"]: t for t in arm1}
    twins = []
    for m, t in masks.items():
        comp = "".join("A" if c == "B" else "B" for c in m)
        if comp in masks and m < comp:
            twins.append(t["iota_A"] + masks[comp]["iota_A"] - 1)
    if twins:
        tw = np.array(twins)
        bs = [rng.choice(tw, len(tw)).mean() for _ in range(2000)]
        summary["H3_twins"] = {"n_pairs": len(tw), "mean_D": float(tw.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}

    sea = [t for t in arm1 if t["seaworthy"]]
    summary["H4_dissociation"] = {"n_seaworthy": len(sea),
                                  "frac_gap_gt_0.5": (sum(abs(t["iota_A"] - t["phi"]) > 0.5 for t in sea) / len(sea)) if sea else None}

    h5 = {"worse_parent_ppl": worse_ppl}
    for o in ("forward", "backward"):
        ps = [tb[f"arm1/{o}/k{k:02d}"]["ppl"] for k in range(1, NP) if f"arm1/{o}/k{k:02d}" in tb]
        if ps:
            h5[f"max_ppl_{o}"] = max(ps)
    wins = 0
    for j in range(1, 10):
        a = j / 10
        k = max(1, min(NP - 1, round(NP * a)))
        plank = [tb[f"arm1/{o}/k{k:02d}"]["ppl"] for o in ORDERS if f"arm1/{o}/k{k:02d}" in tb]
        interp = tb.get(f"arm3/a{round(a, 1)}")
        if plank and interp:
            wins += int(interp["ppl"] < float(np.mean(plank)))
    h5["interp_lower_count_of_9"] = wins
    summary["H5_seaworthiness"] = h5
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    (OUT / "table.json").write_text(json.dumps(table))
    log("analysis written", json.dumps(summary)[:3000])


if __name__ == "__main__":
    main()
