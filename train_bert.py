"""Fine-tune a Turkish BERT (BERTurk, dbmdz/bert-base-turkish-cased) to tag event requests.

Each word is labeled at its first sub-token; the other sub-tokens are ignored in the loss (-100). Three ways to turn
the model's scores into tags:
- softmax: the best tag of each word on its own (argmax), as in a plain token classifier.
- softmax + BIO rules: the same scores, decoded with Viterbi where transitions that break IOB2 (O -> I-DATE) are
  forbidden. No retraining; this is the constrained decoding of week 6.
- --crf: a linear-chain CRF layer on top of BERTurk (learned transition scores, trained with the CRF loss
  log Z - gold score, decoded with Viterbi). Its decoding also forbids IOB2-breaking transitions.

AdamW with linear warmup, bf16 autocast, batch 16. The epoch with the best dev span F1 is kept; then the dev set and
both test sets are scored once. Writes results/<tag>.json and results/predictions/<tag>_<decoder>.json; the weights
are saved to artifacts/<tag>.pt with --save.

Usage: python train_bert.py [--model NAME] [--crf] [--seed N] [--epochs N] [--save]
Requires torch with a CUDA GPU.
"""

import argparse
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from common import ROOT, TAGS, bio_valid, read_jsonl, span_scores

TAG_ID = {tag: i for i, tag in enumerate(TAGS)}
SPLITS = ("dev", "codex_test", "biplan_test")
LR, HEAD_LR, BATCH, WARMUP, MAX_LEN = 3e-5, 1e-3, 16, 0.1, 128
# allowed[i, j] is True when tag j may follow tag i under IOB2; allowed_start[j] when tag j may start a request.
ALLOWED = torch.tensor([[bio_valid(a, b) for b in TAGS] for a in TAGS])
ALLOWED_START = torch.tensor([bio_valid("O", b) for b in TAGS])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def viterbi(emissions, transitions, start):
    """Best tag path for one request. emissions [words, tags], transitions [tags, tags], start [tags]; all scores
    are added (log space). Forbidden moves are -inf."""
    score = start + emissions[0]
    back = []
    for emission in emissions[1:]:
        total = score.unsqueeze(1) + transitions  # [previous, next]
        score, pointer = total.max(dim=0)
        score = score + emission
        back.append(pointer)
    path = [int(score.argmax())]
    for pointer in reversed(back):
        path.append(int(pointer[path[-1]]))
    return path[::-1]


class Tagger(nn.Module):
    """Encoder + linear layer giving one score per tag for each word; optionally a CRF on top."""

    def __init__(self, name, crf):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name)
        self.head = nn.Linear(self.encoder.config.hidden_size, len(TAGS))
        self.crf = crf
        if crf:
            self.transitions = nn.Parameter(torch.zeros(len(TAGS), len(TAGS)))
            self.start = nn.Parameter(torch.zeros(len(TAGS)))
            self.end = nn.Parameter(torch.zeros(len(TAGS)))

    def word_scores(self, ids, mask, first):
        """Scores [batch, words, tags] read at each word's first sub-token; first is [batch, words] positions."""
        hidden = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        hidden = torch.gather(hidden, 1, first.unsqueeze(-1).expand(-1, -1, hidden.size(-1)))
        return self.head(hidden).float()

    def loss(self, scores, tags, word_mask):
        if not self.crf:
            return F.cross_entropy(scores[word_mask], tags[word_mask])
        return self.crf_nll(scores, tags, word_mask).mean()

    def crf_nll(self, scores, tags, word_mask):
        """-log P(gold path | request) = log Z - score(gold), with Z from the forward algorithm (log-sum-exp)."""
        batch, words, _ = scores.shape
        rows = torch.arange(batch, device=scores.device)
        gold = self.start[tags[:, 0]] + scores[rows, 0, tags[:, 0]]
        alpha = self.start + scores[:, 0]  # [batch, tags]
        for i in range(1, words):
            on = word_mask[:, i]
            step = scores[rows, i, tags[:, i]] + self.transitions[tags[:, i - 1], tags[:, i]]
            gold = gold + step * on
            nxt = torch.logsumexp(alpha.unsqueeze(2) + self.transitions, dim=1) + scores[:, i]
            alpha = torch.where(on.unsqueeze(1), nxt, alpha)
        last = word_mask.sum(1) - 1
        gold = gold + self.end[tags[rows, last]]
        return torch.logsumexp(alpha + self.end, dim=1) - gold

    def decode(self, scores, n_words, rules):
        """Tag ids for one request from its word scores [words, tags]."""
        scores = scores[:n_words].cpu()
        if not self.crf and not rules:
            return scores.argmax(-1).tolist()
        forbid = torch.where(ALLOWED, 0.0, -math.inf)
        forbid_start = torch.where(ALLOWED_START, 0.0, -math.inf)
        if not self.crf:
            return viterbi(scores, forbid, forbid_start)
        emissions = scores.clone()
        emissions[-1] += self.end.detach().cpu()
        return viterbi(emissions, self.transitions.detach().cpu() + forbid, self.start.detach().cpu() + forbid_start)


def encode(tokenizer, rows):
    """Sub-token ids, first sub-token position of each word, and tag ids per request."""
    out = []
    for r in rows:
        enc = tokenizer(r["tokens"], is_split_into_words=True, truncation=True, max_length=MAX_LEN)
        first = {}
        for pos, word in enumerate(enc.word_ids()):
            if word is not None and word not in first:
                first[word] = pos
        if len(first) < len(r["tokens"]):
            raise ValueError(f"request too long for {MAX_LEN} sub-tokens: {r['text']!r}")
        out.append({"ids": enc["input_ids"], "first": [first[i] for i in range(len(r["tokens"]))],
                    "tags": [TAG_ID[t] for t in r["tags"]]})
    return out


def collate(items, pad_id):
    n_sub = max(len(e["ids"]) for e in items)
    n_words = max(len(e["first"]) for e in items)
    ids = torch.full((len(items), n_sub), pad_id)
    mask = torch.zeros((len(items), n_sub), dtype=torch.long)
    first = torch.zeros((len(items), n_words), dtype=torch.long)
    tags = torch.zeros((len(items), n_words), dtype=torch.long)
    word_mask = torch.zeros((len(items), n_words), dtype=torch.bool)
    for i, e in enumerate(items):
        ids[i, :len(e["ids"])] = torch.tensor(e["ids"])
        mask[i, :len(e["ids"])] = 1
        first[i, :len(e["first"])] = torch.tensor(e["first"])
        tags[i, :len(e["tags"])] = torch.tensor(e["tags"])
        word_mask[i, :len(e["first"])] = True
    return [t.cuda() for t in (ids, mask, first, tags, word_mask)]


@torch.no_grad()
def predict(model, examples, pad_id, rules):
    model.eval()
    out = []
    for start in range(0, len(examples), BATCH):
        chunk = examples[start:start + BATCH]
        ids, mask, first, _, _ = collate(chunk, pad_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = model.word_scores(ids, mask, first)
        for e, s in zip(chunk, scores):
            out.append([TAGS[i] for i in model.decode(s, len(e["first"]), rules)])
    return out


def fit(model_name, crf, seed, epochs, train_rows, dev_rows):
    """Fine-tune on train_rows and keep the epoch with the best dev span F1.

    Returns (tokenizer, model, history, train_seconds). Also used by learning_curve.py and robustness.py.
    """
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train, dev = encode(tokenizer, train_rows), encode(tokenizer, dev_rows)
    model = Tagger(model_name, crf).cuda()

    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": LR, "weight_decay": 0.01},
                                   {"params": head, "lr": HEAD_LR, "weight_decay": 0.0}])
    steps = math.ceil(len(train) / BATCH) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP * steps), steps)
    rules = not crf
    history, best_f1, best_state = [], -1.0, None
    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train)).tolist()
        losses = []
        for i in range(0, len(order), BATCH):
            ids, mask, first, tags, word_mask = collate([train[j] for j in order[i:i + BATCH]],
                                                        tokenizer.pad_token_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                scores = model.word_scores(ids, mask, first)
            loss = model.loss(scores, tags, word_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        dev_f1 = span_scores([r["tags"] for r in dev_rows], predict(model, dev, tokenizer.pad_token_id, rules))["f1"]
        history.append({"epoch": epoch, "loss": sum(losses) / len(losses), "dev_f1": dev_f1})
        print(f"epoch {epoch}: loss {history[-1]['loss']:.4f}  dev F1 {dev_f1:.3f}", flush=True)
        if dev_f1 > best_f1:
            best_f1, best_state = dev_f1, {k: v.detach().clone() for k, v in model.state_dict().items()}
    train_seconds = time.perf_counter() - start_time
    model.load_state_dict(best_state)
    return tokenizer, model, history, train_seconds


def tag_rows(model, tokenizer, rows, rules):
    """Predicted tags for rows with "tokens" (and "tags", which only fix the length)."""
    return predict(model, encode(tokenizer, rows), tokenizer.pad_token_id, rules)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dbmdz/bert-base-turkish-cased")
    parser.add_argument("--crf", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    tag = f"{args.model.split('/')[-1]}{'_crf' if args.crf else ''}_seed{args.seed}"
    data = {name: read_jsonl(ROOT / "data" / "clean" / f"{name}.jsonl") for name in ("train",) + SPLITS}
    tokenizer, model, history, train_seconds = fit(args.model, args.crf, args.seed, args.epochs, data["train"],
                                                   data["dev"])
    examples = {name: encode(tokenizer, data[name]) for name in SPLITS}

    out = {"model": args.model, "crf": args.crf, "seed": args.seed, "epochs": args.epochs, "lr": LR,
           "batch": BATCH, "train_seconds": train_seconds, "history": history,
           "best_epoch": max(history, key=lambda h: h["dev_f1"])["epoch"], "decoders": {}}
    preds = {}
    decoders = {"crf": True} if args.crf else {"softmax": False, "softmax_bio": True}
    for decoder, use_rules in decoders.items():
        out["decoders"][decoder], preds[decoder] = {}, {}
        for split in SPLITS:
            torch.cuda.synchronize()
            begin = time.perf_counter()
            preds[decoder][split] = predict(model, examples[split], tokenizer.pad_token_id, use_rules)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - begin) * 1000 / len(data[split])
            out["decoders"][decoder][split] = {**span_scores([r["tags"] for r in data[split]], preds[decoder][split]),
                                               "ms_per_request": ms}
            print(f"{decoder:<12} {split:<12} F1 {out['decoders'][decoder][split]['f1']:.3f}")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{tag}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    (ROOT / "results" / "predictions").mkdir(exist_ok=True)
    for decoder, by_split in preds.items():
        path = ROOT / "results" / "predictions" / f"{tag}_{decoder}.json"
        path.write_text(json.dumps(by_split) + "\n", encoding="utf-8")
    if args.save:
        (ROOT / "artifacts").mkdir(exist_ok=True)
        torch.save(model.state_dict(), ROOT / "artifacts" / f"{tag}.pt")
    print(f"wrote results/{tag}.json")


if __name__ == "__main__":
    main()
