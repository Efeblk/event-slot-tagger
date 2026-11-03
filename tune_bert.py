"""Was BERTurk_E's training setting a lucky guess? Search the encoder learning rate and the number of epochs on dev.

BERTurk + CRF layer (BERTurk_E) is trained for every encoder learning rate in LRS and epoch count in EPOCHS, with
seeds 7, 13 and 42; everything else is as in train_bert.py (head 1e-3, batch 16, 10% warmup, best dev epoch kept).
The setting with the best mean dev F1 is chosen; only then are the tests read, as the paired difference to the
reported setting (3e-5, 5 epochs) with the 95% bootstrap interval of the seed means. The reported setting is
retrained too, and its predictions must equal the saved ones (training is deterministic). The reported model does
not change: this search was run after all results were final.

Writes results/tuning.json after every run, so an interrupted search resumes where it stopped.

Usage: python tune_bert.py     (needs a CUDA GPU)
"""

import json
import statistics

from common import ROOT, bootstrap_mean_f1, read_jsonl, span_scores
from train_bert import LR, SPLITS, fit, tag_rows

MODEL = "dbmdz/bert-base-turkish-cased"
LRS = (1e-5, 2e-5, 3e-5, 5e-5)
EPOCHS = (5, 10)
SEEDS = (7, 13, 42)
TESTS = ("codex_test", "opus_test")
OUT = ROOT / "results" / "tuning.json"


def key(lr, epochs):
    return f"lr{lr:g}_ep{epochs}"


def main():
    data = {name: read_jsonl(ROOT / "data" / "clean" / f"{name}.jsonl") for name in ("train",) + SPLITS}
    gold = {name: [r["tags"] for r in data[name]] for name in SPLITS}
    runs = json.loads(OUT.read_text(encoding="utf-8"))["runs"] if OUT.exists() else {}
    for lr in LRS:
        for epochs in EPOCHS:
            for seed in SEEDS:
                name = f"{key(lr, epochs)}_seed{seed}"
                if name in runs:
                    continue
                print(f"== {name}", flush=True)
                tokenizer, model, history, seconds = fit(MODEL, True, seed, epochs, data["train"], data["dev"], lr)
                preds = {split: tag_rows(model, tokenizer, data[split], False) for split in SPLITS}
                runs[name] = {"lr": lr, "epochs": epochs, "seed": seed, "train_seconds": seconds,
                              "best_epoch": max(history, key=lambda h: h["dev_f1"])["epoch"],
                              "dev_f1": span_scores(gold["dev"], preds["dev"])["f1"], "predictions": preds}
                del model
                OUT.write_text(json.dumps({"runs": runs}) + "\n", encoding="utf-8")

    reported = key(LR, 5)
    for seed in SEEDS:
        saved = json.loads((ROOT / "results" / "predictions" / f"bert-base-turkish-cased_crf_seed{seed}_crf.json")
                           .read_text(encoding="utf-8"))
        same = all(saved[split] == runs[f"{reported}_seed{seed}"]["predictions"][split] for split in SPLITS)
        print(f"reported setting, seed {seed}: predictions equal the saved ones: {same}")

    def group(setting):
        return [runs[f"{setting}_seed{seed}"] for seed in SEEDS]

    grid = {}
    for lr in LRS:
        for epochs in EPOCHS:
            rows = group(key(lr, epochs))
            dev = [r["dev_f1"] for r in rows]
            low, high = bootstrap_mean_f1(gold["dev"], [r["predictions"]["dev"] for r in rows])
            grid[key(lr, epochs)] = {"lr": lr, "epochs": epochs, "dev_f1": dev, "dev_mean": statistics.mean(dev),
                                     "dev_low": low, "dev_high": high,
                                     "best_epochs": [r["best_epoch"] for r in rows],
                                     "train_seconds": statistics.mean(r["train_seconds"] for r in rows)}
    chosen = max(grid, key=lambda k: grid[k]["dev_mean"])
    diff_dev = bootstrap_mean_f1(gold["dev"], [r["predictions"]["dev"] for r in group(chosen)],
                                 [r["predictions"]["dev"] for r in group(reported)])
    tests = {}
    for split in TESTS:
        runs_c, runs_r = ([r["predictions"][split] for r in group(s)] for s in (chosen, reported))
        mean_c = statistics.mean(span_scores(gold[split], p)["f1"] for p in runs_c)
        mean_r = statistics.mean(span_scores(gold[split], p)["f1"] for p in runs_r)
        tests[split] = {"chosen_f1": mean_c, "reported_f1": mean_r, "difference": mean_c - mean_r,
                        "interval": bootstrap_mean_f1(gold[split], runs_c, runs_r)}

    out = {"model": MODEL, "seeds": list(SEEDS), "reported": reported, "chosen": chosen,
           "chosen_minus_reported_dev": {"difference": grid[chosen]["dev_mean"] - grid[reported]["dev_mean"],
                                         "interval": diff_dev},
           "tests": tests, "grid": grid, "runs": runs}
    OUT.write_text(json.dumps(out) + "\n", encoding="utf-8")
    for k, g in grid.items():
        print(f"{k:<14} dev {g['dev_mean']:.4f} [{g['dev_low']:.3f}, {g['dev_high']:.3f}]  best epochs {g['best_epochs']}")
    print(f"chosen on dev: {chosen}; reported: {reported}")
    for split, t in tests.items():
        print(f"{split}: chosen {t['chosen_f1']:.4f}, reported {t['reported_f1']:.4f}, "
              f"difference {t['difference']:+.4f} [{t['interval'][0]:+.3f}, {t['interval'][1]:+.3f}]")


if __name__ == "__main__":
    main()
