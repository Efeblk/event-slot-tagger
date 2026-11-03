"""One command that checks the CPU part of this project end to end (no GPU, about a minute).

1. Runs the unit tests (span conversion, scoring, HMM / CRF math, the scorer against seqeval).
2. Reruns prepare_data.py, train_classic.py and evaluate.py, and checks that every file they write has the same
   content as the committed one. Timing fields (train_seconds, ms_per_request) are not compared, and neither are line
   endings or key order. BERTurk is not retrained; evaluate.py scores its saved predictions.
3. Checks that no training or dev request is also a test request (after lowercasing and folding Turkish letters,
   as prepare_data.py does), and that no test request appears in two test sets.
4. Checks that every number in the two README benchmark tables (Codex and Opus test) and in the table of paired
   differences equals the value in results/evaluation.json, rounded as printed.

The committed files are restored afterwards, so the working tree is unchanged. Exit code 0 when everything matches.

Usage: python verify.py
"""

import json
import os
import subprocess
import sys

from common import ROOT, read_jsonl
from prepare_data import key

OUTPUTS = (sorted((ROOT / "data" / "clean").glob("*")) + [ROOT / "results" / "classic.json"]
           + [ROOT / "results" / "predictions" / f"{m}.json"
              for m in ("frequent_tag", "hmm", "hmm_bio", "crf", "crf_no_suffix")]
           + [ROOT / "results" / "evaluation.json", ROOT / "results" / "error_examples.json"])
TIMING = {"train_seconds", "ms_per_request"}
# README benchmark row label -> (key in evaluation.json, True if trained with several seeds)
ROWS = {"Frequent tag": ("frequent_tag", False), "HMM": ("hmm", False),
        "CRF, no suffix features": ("crf_no_suffix", False), "CRF": ("crf", False),
        "BERTurk, softmax": ("berturk_softmax", True), "BERTurk, softmax + BIO rules": ("berturk_softmax_bio", True),
        "BERTurk_E (CRF layer)": ("berturk_crf", True)}
# README paired-difference row label (minus signs read as "-") -> key in evaluation.json ("differences"); BERTurk rows use the seed means
DIFFERENCES = {"HMM - frequent tag": "hmm - frequent_tag", "CRF - HMM": "crf - hmm_bio",
               "CRF - CRF without suffixes": "crf - crf_no_suffix",
               "BERTurk (BIO rules) - CRF": "berturk_softmax_bio - crf",
               "BERTurk: BIO rules - softmax": "berturk_softmax_bio - berturk_softmax",
               "BERTurk: CRF layer - BIO rules": "berturk_crf - berturk_softmax_bio"}


def content(path):
    """Parsed content without timing fields: a JSON value, or a list of them for .jsonl."""
    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k not in TIMING}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [strip(json.loads(line)) for line in text.splitlines() if line.strip()]
    return strip(json.loads(text))


def run(*command):
    print("$ python", " ".join(command), flush=True)
    result = subprocess.run([sys.executable, *command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if result.returncode:
        print(result.stdout[-3000:], result.stderr[-3000:])
    return result.returncode == 0


def readme_tables():
    """The two benchmark tables (Codex, then Opus) and the paired-difference table, as {row label: [cells]}."""
    tables, current = [], None
    for line in (ROOT / "README.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("| Method | Role | Span micro F1") or line.startswith("| Comparison | Difference"):
            current = {}
            tables.append(current)
        elif current is not None and line.startswith("|"):
            cells = [c.strip().strip("*").replace("−", "-") for c in line.strip("|").split("|")]
            if not set(cells[0]) <= set("-"):
                current[cells[0]] = cells[1:]
        else:
            current = None
    return tables


def check_overlap():
    texts = {name: {key(r["text"]) for r in read_jsonl(ROOT / "data" / "clean" / f"{name}.jsonl")}
             for name in ("train", "dev", "codex_test", "opus_test", "biplan_test")}
    tests = ("codex_test", "opus_test", "biplan_test")
    problems = [f"{a} and {b} share {len(texts[a] & texts[b])} requests"
                for i, a in enumerate(("train", "dev") + tests) for b in (("dev",) + tests)[i:]
                if a != b and texts[a] & texts[b]]
    print(f"overlap between train, dev and the three tests: {'none' if not problems else problems}")
    return not problems


def check_readme(evaluation):
    problems, checked = [], 0
    tables = readme_tables()
    for table, split in zip(tables[:2], ("codex_test", "opus_test")):
        for label, (key, seeded) in ROWS.items():
            if seeded:
                s = evaluation["seeds"][key][split]
                values = (s["mean"], s["mean_macro_f1"], s["mean_request_exact"], s["mean_token_accuracy"])
            else:
                s = evaluation["models"][key][split]
                values = (s["f1"], s["macro_f1"], s["request_exact"], s["token_accuracy"])
            for printed, value in zip(table[label][1:], values):
                checked += 1
                if printed != f"{value:.3f}":
                    problems.append(f"README {split} '{label}': printed {printed}, evaluation.json {value:.4f}")
    for label, key in DIFFERENCES.items():
        d = evaluation["differences"][key]["codex_test"]
        if "berturk" in key:
            d = d["seed_mean"]
        printed = tables[2][label]
        expected = (f"{d['diff']:+.3f}", f"[{d['low']:+.3f}, {d['high']:+.3f}]")
        checked += 2
        if tuple(printed[:2]) != expected:
            problems.append(f"README difference '{label}': printed {printed[:2]}, evaluation.json {expected}")
    return checked, problems


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ok = run("-m", "pytest", "tests", "-q")
    saved = {path: path.read_bytes() for path in OUTPUTS}
    expected = {path: content(path) for path in OUTPUTS}
    try:
        ok &= run("prepare_data.py") and run("train_classic.py") and run("evaluate.py")
        changed = [path.relative_to(ROOT).as_posix() for path in OUTPUTS if content(path) != expected[path]]
        print(f"reproduced {len(OUTPUTS) - len(changed)} of {len(OUTPUTS)} committed outputs exactly"
              + (f"; different: {changed}" if changed else ""))
        ok &= not changed
    finally:
        for path, data in saved.items():
            path.write_bytes(data)
    ok &= check_overlap()
    checked, problems = check_readme(json.loads((ROOT / "results" / "evaluation.json").read_text(encoding="utf-8")))
    print(f"README numbers checked against results/evaluation.json: {checked}, mismatches: {len(problems)}")
    for problem in problems:
        print("  " + problem)
    ok &= not problems
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
