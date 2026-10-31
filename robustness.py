"""Attack the trained taggers from two sides: corrupted input and minimal pairs.

1. Perturbed Codex test. Every word is changed in place, so the gold tags stay valid: Turkish lowercase, no Turkish
   letters (ş -> s, ı -> i, ...), no apostrophes (İzmir'de -> İzmirde), ALL CAPS, random typos in 10% / 25% of the
   words (swap two letters, drop one, double one), and "messy" (lowercase + no Turkish letters + no apostrophes +
   10% typos, like a hurried chat message).
2. Contrast set (data/contrast/contrast.jsonl, from make_contrast.py): a pair is right only when the target span gets
   the wanted label in one request and EXCLUDE in the other.

The models are retrained here with the settings and seeds of train_classic.py / train_bert.py (training is
deterministic, so they are the reported models; the clean scores are checked against results/evaluation.json).
Writes results/robustness.json.

Usage: python robustness.py [--no-bert]     (BERTurk needs a CUDA GPU)
"""

import argparse
import json
import random
import statistics

from common import ROOT, read_jsonl, span_scores, spans
from crf import CRF
from hmm import HMM, FrequentTag

SEEDS = (7, 13, 42)
FOLD = str.maketrans("ıİşŞğĞüÜöÖçÇ", "iIsSgGuUoOcC")


def tr_lower(word):
    return word.replace("I", "ı").replace("İ", "i").lower()


def tr_upper(word):
    return word.replace("i", "İ").replace("ı", "I").upper()


def typo(word, rng):
    """One random edit inside a word of 4+ letters: swap two neighbours, drop a letter or double a letter."""
    i = rng.randrange(1, len(word) - 1)
    kind = rng.choice(("swap", "drop", "double"))
    if kind == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if kind == "drop":
        return word[:i] + word[i + 1:]
    return word[:i] + word[i] + word[i:]


def with_typos(tokens, rate, key):
    rng = random.Random(key)  # a string seed is stable across runs
    return [typo(t, rng) if t.isalpha() and len(t) >= 4 and rng.random() < rate else t for t in tokens]


def no_apostrophe(word):
    return word if len(word) == 1 else word.replace("'", "").replace("’", "")


PERTURBATIONS = {
    "clean": lambda tokens, key: tokens,
    "lowercase": lambda tokens, key: [tr_lower(t) for t in tokens],
    "no_turkish_letters": lambda tokens, key: [t.translate(FOLD) for t in tokens],
    "no_apostrophes": lambda tokens, key: [no_apostrophe(t) for t in tokens],
    "all_caps": lambda tokens, key: [tr_upper(t) for t in tokens],
    "typos_10": lambda tokens, key: with_typos(tokens, 0.10, key),
    "typos_25": lambda tokens, key: with_typos(tokens, 0.25, key),
    "messy": lambda tokens, key: with_typos([no_apostrophe(tr_lower(t).translate(FOLD)) for t in tokens], 0.10, key),
}


def perturbed_sets(rows):
    out = {}
    for name, change in PERTURBATIONS.items():
        out[name] = []
        for r in rows:
            tokens = change(r["tokens"], f"{name}:{r['id']}")
            out[name].append({**r, "tokens": tokens, "text": " ".join(tokens)})
    return out


def contrast_scores(rows, preds):
    """Target-span accuracy of each version and of whole pairs, overall and per phenomenon."""
    hit = {}
    for r, p in zip(rows, preds):
        label, start, end = r["target"]
        hit[(r["pair"], r["version"])] = (label, start, end) in spans(p)
    pairs = {r["pair"]: r["phenomenon"] for r in rows}

    def summary(keep):
        chosen = [k for k, ph in pairs.items() if keep(ph)]
        return {"pairs": len(chosen),
                "want": statistics.mean(hit[(k, "want")] for k in chosen),
                "exclude": statistics.mean(hit[(k, "exclude")] for k in chosen),
                "pair": statistics.mean(hit[(k, "want")] and hit[(k, "exclude")] for k in chosen)}

    return {"all": summary(lambda ph: True),
            **{ph: summary(lambda x, ph=ph: x == ph) for ph in sorted(set(pairs.values()))}}


def evaluate(predict_rows, attacked, contrast):
    """Scores of one trained model on every perturbed set and on the contrast set."""
    out = {"perturbed": {}}
    for name, rows in attacked.items():
        scores = span_scores([r["tags"] for r in rows], predict_rows(rows))
        out["perturbed"][name] = {"f1": scores["f1"], "labels": {k: v["f1"] for k, v in scores["labels"].items()}}
    out["contrast"] = contrast_scores(contrast, predict_rows(contrast))
    return out


def classic_models():
    chosen = json.loads((ROOT / "results" / "classic.json").read_text(encoding="utf-8"))["crf_chosen"]
    return {"frequent_tag": FrequentTag(), "hmm": HMM(), "crf": CRF(chosen["c1"], chosen["c2"]),
            "crf_no_suffix": CRF(chosen["c1"], chosen["c2"], suffixes=False)}


def bert_results(train, dev, attacked, contrast):
    from train_bert import fit, tag_rows
    out = {}
    for name, crf in (("berturk_softmax_bio", False), ("berturk_crf", True)):
        out[name] = {}
        for seed in SEEDS:
            tokenizer, model, _, _ = fit("dbmdz/bert-base-turkish-cased", crf, seed, 5, train, dev)
            out[name][seed] = evaluate(lambda rows: tag_rows(model, tokenizer, rows, rules=True), attacked, contrast)
            print(f"{name} seed {seed}: clean F1 {out[name][seed]['perturbed']['clean']['f1']:.3f}", flush=True)
            del model
    return out


def average(by_seed):
    """Mean (and per-seed values) of every number in the per-seed results."""
    first = next(iter(by_seed.values()))
    if isinstance(first, dict):
        return {k: average({s: v[k] for s, v in by_seed.items()}) for k in first}
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        return statistics.mean(by_seed.values())
    return first


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-bert", action="store_true")
    args = parser.parse_args()
    data = {name: read_jsonl(ROOT / "data" / "clean" / f"{name}.jsonl") for name in ("train", "dev", "codex_test")}
    attacked = perturbed_sets(data["codex_test"])
    contrast = read_jsonl(ROOT / "data" / "contrast" / "contrast.jsonl")
    reported = json.loads((ROOT / "results" / "evaluation.json").read_text(encoding="utf-8"))

    out = {"perturbations": list(PERTURBATIONS), "requests": len(data["codex_test"]), "contrast_pairs":
           len(contrast) // 2, "models": {}, "seeds": {}}
    for name, model in classic_models().items():
        model.fit(data["train"])
        out["models"][name] = evaluate(lambda rows: [model.predict(r["tokens"]) for r in rows], attacked, contrast)
        print(f"{name}: clean F1 {out['models'][name]['perturbed']['clean']['f1']:.3f}", flush=True)
    if not args.no_bert:
        for name, by_seed in bert_results(data["train"], data["dev"], attacked, contrast).items():
            out["seeds"][name] = by_seed
            out["models"][name] = average(by_seed)

    for name, result in out["models"].items():  # the retrained models must be the reported ones
        key = "seeds" if name in out["seeds"] else "models"
        expected = (reported["seeds"][name]["codex_test"]["mean"] if key == "seeds"
                    else reported["models"][name]["codex_test"]["f1"])
        assert abs(result["perturbed"]["clean"]["f1"] - expected) < 1e-9, (name, expected)

    (ROOT / "results" / "robustness.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\n{'model':<22}" + "".join(f"{p[:9]:>10}" for p in PERTURBATIONS) + "  contrast pairs")
    for name, result in out["models"].items():
        print(f"{name:<22}" + "".join(f"{result['perturbed'][p]['f1']:>10.3f}" for p in PERTURBATIONS)
              + f"  {result['contrast']['all']['pair']:.3f}")
    print("wrote results/robustness.json")


if __name__ == "__main__":
    main()
