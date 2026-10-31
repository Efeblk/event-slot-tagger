"""How much training data does each tagger need? Train on 10%, 25% and 50% of the training requests and score the
Codex test (100% is the reported result).

Each fraction is drawn three times (seeds 7, 13, 42; for BERTurk the same seed also sets the training run), with the
settings of train_classic.py and train_bert.py: the HMM, the CRF with the chosen c1, c2, and BERTurk decoded with
BIO rules. The dev set stays complete, so BERTurk still keeps its best epoch. Writes results/learning_curve.json.

Usage: python learning_curve.py [--no-bert]     (BERTurk needs a CUDA GPU)
"""

import argparse
import json
import random
import statistics

from common import ROOT, read_jsonl, span_scores
from crf import CRF
from hmm import HMM

FRACTIONS = (0.1, 0.25, 0.5)
SEEDS = (7, 13, 42)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-bert", action="store_true")
    args = parser.parse_args()
    data = {name: read_jsonl(ROOT / "data" / "clean" / f"{name}.jsonl") for name in ("train", "dev", "codex_test")}
    gold = [r["tags"] for r in data["codex_test"]]
    chosen = json.loads((ROOT / "results" / "classic.json").read_text(encoding="utf-8"))["crf_chosen"]
    reported = json.loads((ROOT / "results" / "evaluation.json").read_text(encoding="utf-8"))
    makers = {"hmm": HMM, "crf": lambda: CRF(chosen["c1"], chosen["c2"])}
    if not args.no_bert:
        from train_bert import fit, tag_rows

    out = {"train_requests": len(data["train"]), "fractions": list(FRACTIONS) + [1.0], "models": {}}
    full = {"hmm": [reported["models"]["hmm"]["codex_test"]["f1"]],
            "crf": [reported["models"]["crf"]["codex_test"]["f1"]],
            "berturk_softmax_bio": reported["seeds"]["berturk_softmax_bio"]["codex_test"]["f1"]}
    for name in list(makers) + ([] if args.no_bert else ["berturk_softmax_bio"]):
        points = []
        for fraction in FRACTIONS:
            f1s = []
            for seed in SEEDS:
                train = random.Random(seed).sample(data["train"], round(fraction * len(data["train"])))
                if name in makers:
                    model = makers[name]().fit(train)
                    preds = [model.predict(r["tokens"]) for r in data["codex_test"]]
                else:
                    tokenizer, model, _, _ = fit("dbmdz/bert-base-turkish-cased", False, seed, 5, train, data["dev"])
                    preds = tag_rows(model, tokenizer, data["codex_test"], rules=True)
                    del model
                f1s.append(span_scores(gold, preds)["f1"])
            points.append({"fraction": fraction, "requests": len(train), "f1": f1s, "mean": statistics.mean(f1s)})
            print(f"{name:<20} {fraction:>5.0%}  Codex F1 {statistics.mean(f1s):.3f}", flush=True)
        points.append({"fraction": 1.0, "requests": len(data["train"]), "f1": full[name],
                       "mean": statistics.mean(full[name])})
        out["models"][name] = points
    (ROOT / "results" / "learning_curve.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote results/learning_curve.json")


if __name__ == "__main__":
    main()
