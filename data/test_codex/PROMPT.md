# Prompt given to the AI writer of the Codex test set

The test set was written by a different AI than the training data (OpenAI Codex, model gpt-6-luna), so a model
cannot score well only by copying the training writer's style. Codex received this text and `data/GUIDELINES.md`,
and did not see any training file.

---

Write 300 Turkish requests that a person might type to an event-finding chat assistant in Turkey, with their labels,
following `data/GUIDELINES.md` exactly. Save them to `data/test_codex/codex_test.jsonl`, one JSON object per line:
`{"text": "...", "spans": [["LABEL", "exact substring"], ...]}`.

- Write the way people really type on a phone: many requests lowercase and without Turkish letters, some typos and
  slang, some correctly spelled. Mix short and long requests and questions.
- Cover all seven labels, many cities and districts across Turkey (not only İstanbul), many kinds of events, real
  public performer and team names, prices, group sizes and companions. About 30 requests should exclude something
  (EXCLUDE) and about 20 should ask for nothing specific.
- Copy every span exactly as written, in text order, on whole words, without overlaps.
- Do not read any other file in the repository.
- Write every request yourself, one by one. Do not generate requests with a program, template or loop, and do not
  reuse a sentence frame more than 3 times.
- Write the file with your file-editing tool as UTF-8 and keep the Turkish letters (ç ğ ı ö ş ü İ) wherever a
  request uses them.

A first attempt (2026-10-27) was rejected and deleted without being used: it was built from one sentence template
(`... var mı bütçem ... 2 kişi gideceğiz`) and every Turkish letter had been written as `U`. The two rules above
were added for the second attempt.

The second attempt stopped at its output limit after 99 requests (`codex_test_a.jsonl`). The remaining 200 were
asked for in two batches of 100 with the same text plus one style line each: `codex_test_b.jsonl` (messy phone-chat
style, mostly without Turkish letters) and `codex_test_c.jsonl` (half messy, half correctly spelled, longer requests,
cities outside İstanbul).
