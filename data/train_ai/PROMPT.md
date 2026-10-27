# Prompt given to the AI writers of the training data

Each part file `partNN.jsonl` was written by one Claude Haiku 5.5 agent that received this text, the labeling guide
`data/GUIDELINES.md`, and one focus line from the table at the end.

---

Write 200 Turkish requests that a person might type to an event-finding chat assistant in Turkey, with their
labels, following `data/GUIDELINES.md` exactly. Save them to `data/train_ai/partNN.jsonl`, one JSON object per line:
`{"text": "...", "spans": [["LABEL", "exact substring"], ...]}`.

Rules:
- Write like real users on a phone: mostly lowercase, often without Turkish letters (`kadikoyde`, `aksam`, `ucretsiz`),
  some typos, slang (`bi`, `naber`, `abi`, `ya`, `bişey`, `mı var`), short and long requests, a few questions, a few
  with emojis. About one in four requests may use correct Turkish spelling and capitals.
- Vary everything: cities (not only İstanbul), districts, venues, dates, clock times, prices, group sizes,
  companions, event kinds, performer names, and sentence order. Do not repeat a sentence pattern more than 3 times.
- About 1 request in 10 asks for nothing specific (`spans` is empty), and about 1 in 8 excludes something (EXCLUDE).
- Copy every span exactly as it appears in your text, including your own typos and suffixes, and list spans in text
  order. Spans must cover whole words and must not overlap.
- Use only fictional or public performer, team and venue names; no personal data.

## Focus per part

| Part | Focus |
|---|---|
| part01 | İstanbul districts and both sides of the city; concerts and stand-up |
| part02 | Ankara and Eskişehir; theatre, opera, ballet; students on a budget |
| part03 | İzmir, Bursa, Antalya; festivals, open-air events, summer |
| part04 | Requests with prices and group sizes (per person, total, free) |
| part05 | Requests that exclude kinds, names or places (`olmasın`, `hariç`, `istemiyorum`, `dışında`) |
| part06 | Named performers, bands, comedians, football and basketball teams, plays and films |
| part07 | Clock times and date ranges (`saat 8'den sonra`, `15-20 kasım arası`, `önümüzdeki ay`) |
| part08 | Families and children, couples, friends; museums, exhibitions, workshops, cinema |
| part09 | Very messy chat style: no Turkish letters, typos, slang, run-on requests |
| part10 | Mixed: everything above, longer requests with three or more labels |
