"""Attack the trained taggers from two sides: corrupted input and minimal pairs.

1. Perturbed Codex test. Every word is changed in place, so the gold tags stay valid: Turkish lowercase, no Turkish
   letters (ş -> s, ı -> i, ...), no apostrophes (İzmir'de -> İzmirde), ALL CAPS, random typos in 10% / 25% of the
   words (swap two letters, drop one, double one), and "messy" (lowercase + no Turkish letters + no apostrophes +
   10% typos, like a hurried chat message).
2. Contrast set (data/contrast/contrast.jsonl, from make_contrast.py): a pair is right only when the target span gets
   the wanted label in one request and EXCLUDE in the other.
Every score has a 95% bootstrap interval over requests (pairs for the contrast set); for BERTurk it is the interval of
the mean over 3 seeds, and the change against the clean text is a paired interval.

The models are retrained here with the settings and seeds of train_classic.py / train_bert.py (training is
deterministic, so they are the reported models; the clean scores are checked against results/evaluation.json).
Writes results/robustness.json.

Usage: python robustness.py [--no-bert]     (BERTurk needs a CUDA GPU)
"""

import argparse
import json
import random
import statistics

from common import ROOT, bootstrap_mean, bootstrap_mean_f1, read_jsonl, span_scores, spans
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


def contrast_scores(rows, runs):
    """Target-span accuracy of each version and of whole pairs, overall and per phenomenon, with 95% bootstrap
    intervals over pairs. runs: one prediction list per trained model (seed)."""
    hits = []
    for preds in runs:
        hit = {}
        for r, p in zip(rows, preds):
            label, start, end = r["target"]
            hit[(r["pair"], r["version"])] = (label, start, end) in spans(p)
        hits.append(hit)
    pairs = {r["pair"]: r["phenomenon"] for r in rows}

    def summary(keep):
        chosen = [k for k, ph in pairs.items() if keep(ph)]
        out = {"pairs": len(chosen)}
        for part, right in (("want", lambda h, k: h[(k, "want")]), ("exclude", lambda h, k: h[(k, "exclude")]),
                            ("pair", lambda h, k: h[(k, "want")] and h[(k, "exclude")])):
            values = [[right(h, k) for k in chosen] for h in hits]
            low, high = bootstrap_mean(values)
            out[part] = statistics.mean(v for run in values for v in run)
            out[part + "_low"], out[part + "_high"] = low, high
        return out

    return {"all": summary(lambda ph: True),
            **{ph: summary(lambda x, ph=ph: x == ph) for ph in sorted(set(pairs.values()))}}


def predictions(predict_rows, attacked, contrast):
    """Predicted tags of one trained model on every perturbed set and on the contrast set."""
    return {"perturbed": {name: predict_rows(rows) for name, rows in attacked.items()},
            "contrast": predict_rows(contrast)}


def score(runs, attacked, contrast):
    """F1 on every perturbed set and its change against the clean text (mean over runs, with 95% bootstrap
    intervals; the change is a paired interval on the same drawn requests), and the contrast set scores."""
    gold = [r["tags"] for r in attacked["clean"]]
    clean = [run["perturbed"]["clean"] for run in runs]
    out = {"perturbed": {}}
    for name in attacked:
        preds = [run["perturbed"][name] for run in runs]
        f1s = [span_scores(gold, p)["f1"] for p in preds]
        low, high = bootstrap_mean_f1(gold, preds)
        change_low, change_high = bootstrap_mean_f1(gold, preds, clean)
        out["perturbed"][name] = {"f1": statistics.mean(f1s), "seed_f1": f1s, "low": low, "high": high,
                                  "change": statistics.mean(f1s) - statistics.mean(span_scores(gold, p)["f1"]
                                                                                   for p in clean),
                                  "change_low": change_low, "change_high": change_high}
    out["contrast"] = contrast_scores(contrast, [run["contrast"] for run in runs])
    return out


def classic_models():
    chosen = json.loads((ROOT / "results" / "classic.json").read_text(encoding="utf-8"))["crf_chosen"]
    return {"frequent_tag": FrequentTag(), "hmm": HMM(), "crf": CRF(chosen["c1"], chosen["c2"]),
            "crf_no_suffix": CRF(chosen["c1"], chosen["c2"], suffixes=False)}


def bert_runs(train, dev, attacked, contrast):
    from train_bert import fit, tag_rows
    out = {}
    for name, crf in (("berturk_softmax_bio", False), ("berturk_crf", True)):
        out[name] = []
        for seed in SEEDS:
            tokenizer, model, _, _ = fit("dbmdz/bert-base-turkish-cased", crf, seed, 5, train, dev)
            out[name].append(predictions(lambda rows: tag_rows(model, tokenizer, rows, rules=True), attacked,
                                         contrast))
            print(f"{name} seed {seed} done", flush=True)
            del model
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-bert", action="store_true")
    args = parser.parse_args()
    data = {name: read_jsonl(ROOT / "data" / "clean" / f"{name}.jsonl") for name in ("train", "dev", "codex_test")}
    attacked = perturbed_sets(data["codex_test"])
    contrast = read_jsonl(ROOT / "data" / "contrast" / "contrast.jsonl")
    reported = json.loads((ROOT / "results" / "evaluation.json").read_text(encoding="utf-8"))

    runs = {}
    for name, model in classic_models().items():
        model.fit(data["train"])
        runs[name] = [predictions(lambda rows: [model.predict(r["tokens"]) for r in rows], attacked, contrast)]
    if not args.no_bert:
        runs.update(bert_runs(data["train"], data["dev"], attacked, contrast))

    out = {"perturbations": list(PERTURBATIONS), "requests": len(data["codex_test"]), "contrast_pairs":
           len(contrast) // 2, "seeds": {name: list(SEEDS) for name, r in runs.items() if len(r) > 1}, "models": {}}
    for name, model_runs in runs.items():
        out["models"][name] = score(model_runs, attacked, contrast)
        expected = (reported["seeds"][name]["codex_test"]["mean"] if len(model_runs) > 1
                    else reported["models"][name]["codex_test"]["f1"])
        clean = out["models"][name]["perturbed"]["clean"]["f1"]
        assert abs(clean - expected) < 1e-9, (name, clean, expected)  # the retrained models are the reported ones

    (ROOT / "results" / "robustness.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\n{'model':<22}" + "".join(f"{p[:9]:>10}" for p in PERTURBATIONS) + "  contrast pairs")
    for name, result in out["models"].items():
        print(f"{name:<22}" + "".join(f"{result['perturbed'][p]['f1']:>10.3f}" for p in PERTURBATIONS)
              + f"  {result['contrast']['all']['pair']:.3f}")
    print("wrote results/robustness.json")


if __name__ == "__main__":
    main()
