# Guide for reviewing agents

This file is for an AI agent (or a person) reviewing this repository. It says what the project claims, where the
evidence for each claim is, and how to check it yourself. `README.md` is the full report; this file is the map.

## The project in three lines

COE025 NLP Project 2 (token classification), solo. Task: **Turkish event-request slot filling with negation scope**:
one BIO tag per word for 7 labels (PLACE, DATE, TYPE, NAME, PRICE, PEOPLE, EXCLUDE = 15 tags). The data was built for
this project (AI-written train, tests by two other AI writers + 25 real app requests). Methods go from a frequent-tag
baseline through an own HMM and a CRF to fine-tuned BERTurk with a CRF layer (BERTurk_E, the final model).

## Check it in one command (CPU, about a minute)

```
python -m pip install -r requirements.txt
python verify.py
```

`verify.py` exits 0 only if all of these hold, and leaves the working tree unchanged:

1. The unit tests pass (`tests/test_tagging.py`): tokenizer and span ↔ BIO conversion, exact-span scoring, the HMM
   Viterbi and the CRF loss against the week 6 slide examples, and our scorer against `seqeval` on every saved
   prediction. The CRF-layer test needs `torch` and is skipped without it.
2. `prepare_data.py`, `train_classic.py` and `evaluate.py` rerun from the raw data and reproduce all 14 committed
   outputs exactly (data splits, classic predictions, `results/evaluation.json`); only timing fields may differ.
3. No request is shared between train, dev and the three test sets (after lowercasing and folding Turkish letters).
4. Every number in the two README benchmark tables and the paired-difference table equals `results/evaluation.json`.

CI (`.github/workflows/tests.yml`) runs the same on every push.

The GPU part (BERTurk fine-tuning) is not rerun by `verify.py`; its saved predictions in `results/predictions/` are
scored by `evaluate.py`. To check it on a CUDA GPU: `python train_bert.py --crf --seed 7` rewrites
`results/predictions/bert-base-turkish-cased_crf_seed7_crf.json`; training is deterministic, so `git diff` on that file
should be empty. `tune_bert.py` does this check for all three seeds and prints the result.

## Claims and where the evidence is

| Claim | Evidence | How to check |
|---|---|---|
| The task is not trivial (teacher's bar: baseline ≥ 90% accuracy or ≥ 85% EM) | Frequent tag: 0.750 token accuracy, 16.4% request exact match on the Codex test | `results/evaluation.json` → `models.frequent_tag.codex_test` |
| Final model: BERTurk_E, 0.941 span F1 / 79.6% EM on Codex, 0.942 on Opus | Mean of seeds 7, 13, 42 with bootstrap intervals | `evaluation.json` → `seeds.berturk_crf` |
| Tests were not used to choose anything | Model and settings are chosen on dev; the README reports where dev and tests disagree (the "Which pre-trained model" section) instead of re-choosing | `evaluation.json` → `dev`; `results/tuning.json` (learning-rate and epoch search, dev only) |
| No leakage between splits | Dedup after folding in `prepare_data.py` | `verify.py` step 3 |
| Our scorer is standard exact-span F1 | Same P/R/F1 as `seqeval` on all 92 model/split pairs | `tests/test_tagging.py::test_span_scores_match_seqeval_on_every_saved_prediction` |
| Each addition is measured against its own baseline, with paired 95% intervals | Paired bootstrap over requests, same draws for both models | `evaluation.json` → `differences`; `common.py::bootstrap_mean_f1` |
| The HMM and the CRF layer are implemented correctly | Worked examples of the week 6 slides ("can fish": joint 0.1344; Z = 16, loss = ln 2) | `tests/test_tagging.py` |
| EXCLUDE needs context: the HMM gets 0 of 140 minimal pairs, BERTurk_E 0.99 | `make_contrast.py` → `data/contrast/`, scored in `robustness.py` | `results/robustness.json` |
| Existing Turkish slot data has no exclusion slot | MASSIVE tr-TR: 40 candidates, 4 real exclusions, all labeled like wanted items | `massive_check.py`, `results/massive_negation.json` |
| Test labels were reviewed | 23 corrections to the Codex test, each with a reason; originals unchanged | `data/test_codex/fixes.json` |
| Data cleaning drops bad lines | Every span must match whole words in order | `data/clean/stats.json` → `dropped_lines` |

## Protocol (what was decided when)

1. Guide written (`data/GUIDELINES.md`), training data generated (`data/train_ai/PROMPT.md`).
2. Codex test written by a different AI from the guide only, reviewed by hand (`fixes.json`).
3. All models trained; settings and model chosen on dev; tests scored once.
4. Added after the results were final, never used for a choice: Opus test, stress tests (`robustness.py`), contrast
   set, learning curve, the other pre-trained encoders, the learning-rate search (`tune_bert.py`).

## Known limitations (stated in the README, not hidden)

- Training and the two main tests are AI-written, not collected from users; only the 25 Bi' Plan requests are real,
  so their intervals are wide.
- One annotator reviewed the labels; there is no inter-annotator agreement.
- Dev is from the same writer as train, so it is too optimistic and cannot separate the strong encoders.
- One gold-label inconsistency (excluded name + event word, `Galatasaray maçı olmasın`) is documented in the README
  and was deliberately not fixed after the tests were scored.
- No LLM-prompting baseline; no normalization of spans into values (the project is span tagging only).

## Layout

`common.py` (tokenizer, BIO, scoring, bootstrap) → `prepare_data.py` (raw `data/*/` → `data/clean/`) →
`train_classic.py` (`hmm.py`, `crf.py`) and `train_bert.py` → `results/predictions/` → `evaluate.py` →
`results/evaluation.json` → `plot_results.py` → `results/figures/`. Extra analyses: `robustness.py`,
`make_contrast.py`, `learning_curve.py`, `tune_bert.py`, `massive_check.py`. `project1/` is the earlier project,
unchanged and not part of this review.
