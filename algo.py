"""Soft-weight inference for code API unlearning.

Steers a causal code LM away from a deprecated API call and toward its
replacement, at test time, without touching model weights.

    Step 1  v_steer = mean h(x + y_pos) - mean h(x + y_neg)
    Step 2  gate a = sigmoid(MLP(h)), trained to minimise
            1 - cos(h_neg + a * v_steer, h_pos)
    Step 3  at inference, h'(x) = h(x) + t * a(h(x)) * v_steer

Reused from MLLMEraser (https://anonymous.4open.science/r/MLLMEraser-B41D/),
vendored under ./MLLMEraser-B41D. Provenance is marked inline as
[MLLMEraser: <file>:<lines>].
"""

import argparse
import json
import logging
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- data

def load_json(path, limit=None):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if limit is not None:
        data = data[:limit]
    logger.info(f"{path}: {len(data)} samples")
    return data


def build_prompt(code_context):
    prompt_text = (
        f"Complete and output the next line for the following Python function:\n"
        f"```python\n"
        f"{code_context}"
    )
    return prompt_text


# ------------------------------------------------------- hidden extraction

class HiddenExtractor:
    """Last-token hidden state at one layer, batched.

    [MLLMEraser: Extractor.py:80-106] EmbeddingExtractor.extract_embeddings_text,
    specialised to a single layer and a text-only causal LM.

    `layer` is the index of the decoder block whose *output* we take, so that it
    matches the block the steering hook is attached to. hidden_states[i] is the
    input of block i, hence the +1. (The original indexes hidden_states[i] but
    applies the steering inside block i -- off by one.)
    """

    def __init__(self, model, tokenizer, device, max_len):
        self.model, self.tok, self.device, self.max_len = model, tokenizer, device, max_len

    @torch.no_grad()
    def last_token(self, texts, layer, batch_size):
        out = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"extract L{layer}"):
            enc = self.tok(
                texts[i : i + batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            ).to(self.device)
            # Extraction needs decoder activations, not vocabulary logits or a KV cache.
            hs = self.model.model(
                **enc, output_hidden_states=True, use_cache=False
            ).hidden_states
            out.append(hs[layer + 1][:, -1, :].detach().float().cpu())
            del hs
        h = torch.cat(out, dim=0)
        logger.info(f"H shape: {tuple(h.shape)}")
        return h


# ------------------------------------------------------ Step 1: v_steer

def build_steer_vector(ex, data, layer, batch_size, gate_train_input):
    """v_steer = v_rep - v_dep, and the pair of activation banks the gate trains on."""
    probes = [build_prompt(r["probing input"]) for r in data]
    h_neg = ex.last_token([p + r["y_neg"] for p, r in zip(probes, data)], layer, batch_size)
    h_pos = ex.last_token([p + r["y_pos"] for p, r in zip(probes, data)], layer, batch_size)

    v_steer = h_pos.mean(0) - h_neg.mean(0)
    logger.info(f"||v_steer|| = {v_steer.norm():.4f}")

    # The gate sees x+y_neg at train time but only x at inference. `probe` trains
    # it on x instead, removing that mismatch.
    h_in = ex.last_token(probes, layer, batch_size) if gate_train_input == "probe" else h_neg
    return h_in, h_neg, h_pos, v_steer


# --------------------------------------------------------- Step 2: gate

class Gate(nn.Module):
    """h -> a in (0, 1): how much of v_steer this input needs."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, h):
        return torch.sigmoid(self.net(h))


def train_gate(h_in, h_neg, h_pos, v_steer, device, args):
    gate = Gate(h_in.shape[1], args.gate_hidden).to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=args.lr)
    loader = DataLoader(
        TensorDataset(h_in, h_neg, h_pos), batch_size=args.gate_bs, shuffle=True
    )
    v = v_steer.to(device)

    for epoch in range(args.epochs):
        tot = n = 0.0
        a_sum = 0.0
        for hi, hn, hp in loader:
            hi, hn, hp = hi.to(device), hn.to(device), hp.to(device)
            a = gate(hi)
            loss = (1 - F.cosine_similarity(hn + a * v, hp, dim=-1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * hi.size(0)
            a_sum += a.sum().item()
            n += hi.size(0)
        logger.info(f"epoch {epoch + 1}/{args.epochs}  loss {tot / n:.4f}  mean a {a_sum / n:.3f}")
    return gate.eval()


# ------------------------------------------------------ Step 3: steering

class SteeringHook:
    """h' = h + t * a(h_last) * v_steer, on the prefill pass only.

    [MLLMEraser: qwen_steering.py:108-116] Same injection rule as
    AlphaQwen2_5_VLDecoderLayer.forward -- fires only while shape[1] > 1 (so the
    prompt is steered and the KV cache carries the effect into decoding), derives
    the coefficient from the last prompt token, and broadcasts one vector over
    every position. Moved from a DecoderLayer subclass into a forward hook so it
    does not depend on the decoder-layer forward signature.
    """

    def __init__(self, gate, v_steer, strength):
        self.gate, self.v, self.t = gate, v_steer, strength
        self.enabled = False
        self.last_a = None  # First prompt-pass coefficients; reset for every generate batch.

    def __call__(self, module, args, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        if not self.enabled or hs.shape[1] <= 1:
            return output
        with torch.no_grad():
            a = self.gate(hs[:, -1, :].float())              # [B, 1]
            if self.last_a is None:
                self.last_a = a.detach().flatten()
            delta = (self.t * a * self.v).unsqueeze(1)       # [B, 1, d]
        hs = hs + delta.to(hs.dtype)
        return (hs,) + output[1:] if is_tuple else hs


# ----------------------------------------------------------- evaluation

def _exact_re(name):
    return re.compile(r"(?<![\w.])" + re.escape(name) + r"\s*\(")


def _suffix_re(full):
    """numpy.prod -> matches np.prod( / numpy.prod(, but not bare prod(."""
    return re.compile(r"(?<![\w])(?:[A-Za-z_]\w*\.)+" + re.escape(full.split(".")[-1]) + r"\s*\(")


def api_patterns(rec):
    rep = [_exact_re(rec["expected call"]), _suffix_re(rec["replacement api"])]
    alias = rec.get("alias dict") or {}
    deprecated = set(rec["deprecated api"])
    dep = [_suffix_re(d) for d in deprecated]
    dep += [_exact_re(k) for k, v in alias.items() if v in deprecated]
    return rep, dep


@torch.no_grad()
def evaluate(model, tok, data, device, args, tag, hook):
    rep_hits = dep_hits = 0
    samples = []
    gate_scores = []
    for i in tqdm(range(0, len(data), args.gen_bs), desc=f"gen [{tag}]"):
        batch = data[i : i + args.gen_bs]
        enc = tok(
            [build_prompt(r["probing input"]) for r in batch],
            padding=True,
            truncation=True,
            max_length=args.max_len,
            return_tensors="pt",
        ).to(device)
        hook.last_a = None  # Prevent coefficients leaking between batches or modes.
        out = model.generate(
            **enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id
        )
        # [MLLMEraser: MLLMU_eval_steering.py:118-119]
        out = out[:, enc.input_ids.shape[-1] :]
        a_values = (
            hook.last_a.cpu().tolist() if hook.last_a is not None else [None] * len(batch)
        )
        for offset, (rec, ids, a) in enumerate(zip(batch, out, a_values)):
            text = tok.decode(ids, skip_special_tokens=True)
            rep_pat, dep_pat = api_patterns(rec)
            r = any(p.search(text) for p in rep_pat)
            d = any(p.search(text) for p in dep_pat)
            rep_hits += r
            dep_hits += d
            score = {
                "sample_index": i + offset,
                "id": rec.get("id"),
                "function": rec["function"],
                "a": a,
                "steering_scale": a * hook.t if a is not None else 0.0,
                "rep": r,
                "dep": d,
            }
            gate_scores.append(score)
            if len(samples) < args.n_dump:
                samples.append({
                    **score,
                    "probing input": rec["probing input"],
                    "expected": rec["expected call"],
                    "rep": r,
                    "dep": d,
                    "generate": text,
                })

    n = len(data)
    res = {
        "n": n,
        "repAPI_count": rep_hits,
        "depAPI_count": dep_hits,
        "repAPI": rep_hits / n,
        "depAPI": dep_hits / n,
    }
    logger.info(
        f"[{tag}] n={n}  repAPI={rep_hits}/{n} ({res['repAPI']:.3f})  "
        f"depAPI={dep_hits}/{n} ({res['depAPI']:.3f})"
    )
    return res, samples, gate_scores


# ------------------------------------------------------------------ main

def parse_args():
    p = argparse.ArgumentParser(description="Soft-weight steering for code API unlearning.")
    p.add_argument("--model_id", default="deepseek-ai/deepseek-coder-1.3b-instruct")
    p.add_argument("--data_dir", default="data/deepseek")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--layer", type=int, default=12, help="decoder block to steer (0-indexed)")
    p.add_argument("--strength", type=float, default=1.0, help="t in h + t*a*v_steer")
    p.add_argument("--gate_train_input", choices=["neg", "probe"], default="neg")
    p.add_argument("--n_forget", type=int, default=None, help="cap on D_forget size")
    p.add_argument("--max_test", type=int, default=200)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=48)
    p.add_argument("--extract_bs", type=int, default=8)
    p.add_argument("--gen_bs", type=int, default=8)
    p.add_argument("--gate_bs", type=int, default=64)
    p.add_argument("--gate_hidden", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_dump", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.model_id)
    # [MLLMEraser: Extractor.py:39-42] left padding so [:, -1, :] is the real last
    # token; truncation from the left keeps the tail of the code prefix, which is
    # what the completion continues from.
    tok.padding_side = "left"
    tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device).eval()
    layers = model.model.layers
    assert 0 <= args.layer < len(layers), f"--layer must be in [0, {len(layers)})"

    ex = HiddenExtractor(model, tok, device, args.max_len)

    # Step 1
    forget = load_json(os.path.join(args.data_dir, "D_forget.json"), args.n_forget)
    h_in, h_neg, h_pos, v_steer = build_steer_vector(
        ex, forget, args.layer, args.extract_bs, args.gate_train_input
    )
    torch.save(v_steer, os.path.join(args.out_dir, "v_steer.pt"))

    # Step 2
    gate = train_gate(h_in, h_neg, h_pos, v_steer, device, args)
    torch.save(gate.state_dict(), os.path.join(args.out_dir, "gate.pt"))

    # Step 3
    hook = SteeringHook(gate, v_steer.to(device), args.strength)
    handle = layers[args.layer].register_forward_hook(hook)

    results = {"args": vars(args)}
    dumps = {}
    scores = {}
    for name in ["D_test_U_dep", "D_test_U_nondep"]:
        test = load_json(os.path.join(args.data_dir, f"{name}.json"), args.max_test)
        for tag, on in [("baseline", False), ("steered", True)]:
            hook.enabled = on
            res, samples, gate_scores = evaluate(
                model, tok, test, device, args, f"{name}/{tag}", hook
            )
            results[f"{name}/{tag}"] = res
            dumps[f"{name}/{tag}"] = samples
            scores[f"{name}/{tag}"] = gate_scores
    handle.remove()

    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.out_dir, "samples.json"), "w", encoding="utf-8") as f:
        json.dump(dumps, f, indent=2)
    with open(os.path.join(args.out_dir, "gate_scores.json"), "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    logger.info(f"wrote {args.out_dir}/results.json, samples.json and gate_scores.json")


if __name__ == "__main__":
    main()
