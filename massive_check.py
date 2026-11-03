"""Does an existing Turkish slot-filling dataset already cover exclusions? Check MASSIVE 1.1 (tr-TR).

Lists every MASSIVE request in which a slot is followed (directly or after one word) by a negation or exclusion cue
(hariç, dışında, olmasın, değil, istemiyorum, olmayan, yerine), and counts the slot types. Which of these requests
really exclude something from what the user asks for is read by hand (README, "Related work").
Writes results/massive_negation.json.

MASSIVE is not part of this repo: download amazon-massive-dataset-1.1.tar.gz from
https://github.com/alexa/massive and pass the path of 1.1/data/tr-TR.jsonl.

Usage: python massive_check.py PATH/TO/tr-TR.jsonl
"""

import json
import re
import sys
from collections import Counter

from common import ROOT, read_jsonl

CUE = r"(?:hariç|haric|dışında|disinda|olmasın|olmasin|değil|degil|istemiyorum|olmayan|yerine)"
SLOT_THEN_CUE = re.compile(r"\[(\w+) : [^\]]+\]\s*(?:\S+\s+)?" + CUE)
SLOT = re.compile(r"\[(\w+) : ")


def main():
    rows = read_jsonl(sys.argv[1])
    hits = [r for r in rows if SLOT_THEN_CUE.search(r["annot_utt"])]
    out = {"requests": len(rows),
           "slot_types": len(Counter(s for r in rows for s in SLOT.findall(r["annot_utt"]))),
           "slot_then_cue": len(hits),
           "candidates": [{"id": r["id"], "partition": r["partition"], "intent": r["intent"],
                           "annot_utt": r["annot_utt"]} for r in hits]}
    (ROOT / "results" / "massive_negation.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n",
                                                            encoding="utf-8")
    print(f"{out['requests']} requests, {out['slot_types']} slot types, {out['slot_then_cue']} with a slot before a "
          "negation cue (listed in results/massive_negation.json)")


if __name__ == "__main__":
    main()
