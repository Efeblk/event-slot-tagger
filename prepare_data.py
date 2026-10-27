"""Turn the labeled request files into tokenized BIO splits under data/clean/.

Inputs ({"text", "spans"} per line, see data/GUIDELINES.md):
- data/train_ai/part*.jsonl: AI-written training requests (Claude Haiku), split into train and dev.
- data/test_codex/codex_test_*.jsonl: test requests written by a different AI (OpenAI Codex). They were reviewed
  line by line against the guide; the corrected labels are in data/test_codex/fixes.json and replace the original
  spans here, so both versions stay visible.
- data/test_biplan/golden_raw.jsonl: the Turkish requests of the Bi' Plan golden set, labeled by hand.

A line whose spans do not fit its text is dropped and counted, never repaired silently. Texts are compared after
lowercasing, folding Turkish letters to ASCII and removing punctuation, so "Kadıköy'de konser" and "kadikoyde konser"
count as the same request. Duplicates are dropped, then train and dev rows that match a test request.
Dev takes 15% of each part (fixed seed). Writes data/clean/{train,dev,codex_test,biplan_test}.jsonl and stats.json.

Usage: python prepare_data.py
"""

import json
import random
import re
from collections import Counter

from common import ROOT, read_jsonl, spans, to_bio, write_jsonl

FOLD = str.maketrans("çğıöşüâîûÇĞİÖŞÜÂÎÛ", "cgiosuaiuCGIOSUAIU")
DEV_SHARE = 0.15
SEED = 13


def key(text):
    return re.sub(r"[\W_]+", " ", text.translate(FOLD).lower()).strip()


def load(path, prefix, fixes=None):
    """Tagged rows of one file and the list of errors for lines that were dropped. fixes maps (file stem, line
    number) to corrected spans."""
    rows, errors = [], []
    for number, obj in enumerate(read_jsonl(path), 1):
        obj["spans"] = (fixes or {}).get((path.stem, number), obj.get("spans"))
        try:
            tokens, tags = to_bio(obj["text"], obj["spans"])
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{path.name}:{number}: {error}")
            continue
        rows.append({"id": f"{prefix}:{number}", "text": obj["text"], "tokens": tokens, "tags": tags})
    return rows, errors


def dedupe(rows, banned=()):
    """Drop rows whose key is in banned or was already seen. Return (kept, dropped)."""
    seen, kept = set(banned), []
    for row in rows:
        if key(row["text"]) not in seen:
            seen.add(key(row["text"]))
            kept.append(row)
    return kept, len(rows) - len(kept)


def label_counts(rows):
    return dict(Counter(label for r in rows for label, _, _ in spans(r["tags"])).most_common())


def main():
    stats, errors = {}, []
    tests = {}
    fixes = {(f["file"], f["line"]): f["spans"]
             for f in json.loads((ROOT / "data" / "test_codex" / "fixes.json").read_text(encoding="utf-8"))}
    sources = {"codex_test": sorted((ROOT / "data" / "test_codex").glob("codex_test_*.jsonl")),
               "biplan_test": [ROOT / "data" / "test_biplan" / "golden_raw.jsonl"]}
    for name, paths in sources.items():
        rows, bad = [], []
        for path in paths:
            part_rows, part_bad = load(path, f"{name.split('_')[0]}:{path.stem}", fixes)
            rows, bad = rows + part_rows, bad + part_bad
        errors += bad
        tests[name], duplicates = dedupe(rows)
        stats[name] = {"lines": len(rows) + len(bad), "bad_spans": len(bad), "duplicates": duplicates}
    stats["codex_test"]["hand_fixes"] = len(fixes)
    test_keys = {key(r["text"]) for rows in tests.values() for r in rows}

    rng = random.Random(SEED)
    train, dev = [], []
    for path in sorted((ROOT / "data" / "train_ai").glob("part*.jsonl")):
        rows, bad = load(path, f"ai:{path.stem}")
        errors += bad
        rng.shuffle(rows)
        cut = round(len(rows) * DEV_SHARE)
        dev += rows[:cut]
        train += rows[cut:]
        stats[path.stem] = {"lines": len(rows) + len(bad), "bad_spans": len(bad)}
    train, train_dropped = dedupe(train, test_keys)
    dev, dev_dropped = dedupe(dev, test_keys | {key(r["text"]) for r in train})
    stats["train"] = {"duplicates_or_test_matches_dropped": train_dropped}
    stats["dev"] = {"duplicates_or_train_test_matches_dropped": dev_dropped}

    for name, rows in (("train", train), ("dev", dev), *tests.items()):
        write_jsonl(ROOT / "data" / "clean" / f"{name}.jsonl", rows)
        stats[name] = {**stats.get(name, {}), "requests": len(rows), "tokens": sum(len(r["tokens"]) for r in rows),
                       "spans": label_counts(rows)}
    stats["dropped_lines"] = errors
    (ROOT / "data" / "clean" / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
                                                        encoding="utf-8")
    for name in ("train", "dev", *tests):
        print(f"{name:<12} {stats[name]['requests']:>5} requests  {stats[name]['spans']}")
    print(f"{len(errors)} lines dropped for bad spans (listed in data/clean/stats.json)")


if __name__ == "__main__":
    main()
