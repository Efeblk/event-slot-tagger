# İsmet Efe Balık (2509011061, GitHub: Efeblk)

Solo project. Project 2 parts:

- Chose the task: token classification of Turkish event requests with seven labels (place, date, event type, name,
  price, people, excluded), the request parser my app Bi' Plan needs.
- Wrote the labeling guide and the prompts for the AI-written data.
- Built the dataset: 2,000 training requests written by Claude Haiku agents, a 299-request test set written by a
  different AI (OpenAI Codex), and the 25 Turkish requests of the Bi' Plan golden set. Rejected a first Codex attempt
  that was template-made, and kept the label fixes of the Codex test separate from its original files.
- Wrote the cleaning script (span checks, deduplication with Turkish-letter folding, test-leak removal).
- Implemented and compared the frequent-tag baseline, an HMM with Viterbi written from scratch, a CRF with Turkish
  suffix features, and BERTurk with softmax, BIO-rule and CRF-layer decoding (3 seeds each).
- Evaluated with exact-span F1, bootstrap intervals, paired comparisons, slices (no Turkish letters, suffixed places),
  IOB2 breaks, speed, and an error analysis.

I used AI coding assistants (Claude) to help write the code, review the Codex test labels against the guide, and
write this README.
