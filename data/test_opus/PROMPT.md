# Prompt given to the AI writer of the Opus test set

The second test set was written by a third AI writer (Claude Opus, model claude-opus-5-5) on 2026-11-02, after every
model and every Codex/Bi' Plan result was final. It ran as a fresh agent: it did not see this repository's training
data, the other test sets, or any earlier conversation. It was allowed to read only `data/GUIDELINES.md`. The task
text is the Codex prompt (`data/test_codex/PROMPT.md`), with the three-batch split of the Codex history asked for up
front:

---

Write 300 Turkish requests that a person might type to an event-finding chat assistant in Turkey, with their labels,
following data/GUIDELINES.md exactly. Save them as three files of 100 lines each, one JSON object per line:
`{"text": "...", "spans": [["LABEL", "exact substring"], ...]}`

- data/test_opus/opus_test_a.jsonl: mixed style, like real phone typing (many lowercase, some without Turkish
  letters, some typos and slang, some correctly spelled). Short and long requests and questions.
- data/test_opus/opus_test_b.jsonl: messy phone-chat style, mostly without Turkish letters.
- data/test_opus/opus_test_c.jsonl: half messy, half correctly spelled, longer requests, cities outside İstanbul.

Rules:
- Cover all seven labels, many cities and districts across Turkey (not only İstanbul), many kinds of events, real
  public performer and team names, prices, group sizes and companions. In each file, about 10 requests should exclude
  something (EXCLUDE) and about 7 should ask for nothing specific (no spans).
- Copy every span exactly as written in the text, in text order, on whole words, without overlaps. Labels must
  follow the guide's rules on span boundaries and suffixes.
- Write every request yourself, one by one. Do not generate requests with a program, template or loop, and do not
  reuse a sentence frame more than 3 times.
- Write the files with your file-writing tool as UTF-8 and keep the Turkish letters (ç ğ ı ö ş ü İ) wherever a
  request uses them.

---

All 300 lines were then reviewed against the guide; no label was changed, so there is no fixes file. The writer
reported the cases the guide leaves open and how it labeled them: a city followed by a district is two PLACE spans
(`Denizli'de Pamukkale'de`), short team names (`fb`, `gs`, `bjk`, `ts`) are NAME, `uygun fiyatli` is PRICE, an
excluded place keeps its suffix (`Çankaya'da değil` → [Çankaya'da] EXCLUDE), and two companions in a row are two
PEOPLE spans. It wrote no excluded team match (`... maçı istemiyorum`), the one case where the guide is unclear.
