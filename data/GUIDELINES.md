# Labeling guide: Turkish event requests

Every request is one line a user types to an event-finding assistant, for example
`bu akşam kadıköyde en fazla 500 TL stand-up`. Each request is stored as

```json
{"text": "bu akşam kadıköyde en fazla 500 TL stand-up",
 "spans": [["DATE", "bu akşam"], ["PLACE", "kadıköyde"], ["PRICE", "en fazla 500 TL"], ["TYPE", "stand-up"]]}
```

`spans` lists `[label, exact substring]` in the order the substrings appear in the text. A substring must be
copied character for character (same case, same typos, same missing Turkish letters) and must start and end on
whole words. Spans never overlap. Words outside every span are not labeled.

Words are split on spaces and punctuation, but an apostrophe or hyphen inside a word keeps it one word:
`İstanbul'da`, `stand-up`, `21.00'den` are single words.

## Labels

| Label | What it marks | Examples (the span is in brackets) |
|---|---|---|
| PLACE | a city, district, neighbourhood, venue or side of a city, **with its suffix** | [kadıköyde], [İstanbul'da], [Avrupa yakasında], [Zorlu PSM'de], [besiktasta] |
| DATE | when: day, date, part of day, clock time, or a range of these | [bu akşam], [yarın], [hafta sonu], [ekim sonunda], [cumartesi 20.00'den sonra] |
| TYPE | the kind of event, with words that narrow the kind | [konser], [caz konseri], [açık hava konseri], [stand-up], [çocuk tiyatrosu], [maç] |
| NAME | a specific performer, band, team, play, film or festival | [Duman], [Cem Yılmaz], [Fenerbahçe], [Hamlet], [Zeytinli Rock Festivali] |
| PRICE | a price limit or price word, with the words that make it a limit | [en fazla 500 TL], [500 tl altı], [ücretsiz], [bedava], [ucuz], [kişi başı en fazla 600 TL] |
| PEOPLE | how many people, or who comes along | [iki kişi], [4 kişilik], [arkadaşlarla], [sevgilimle], [çocuğumla], [tek başıma] |
| EXCLUDE | a kind, name or place the user does **not** want; only the excluded word(s) | konser olmasın → [konser]; tiyatro hariç → [tiyatro]; stand-up istemiyorum → [stand-up] |

## Boundary rules

1. **Suffixes stay inside the span.** In Turkish the place word carries the case ending: `kadıköyde` is one word and
   the whole word is PLACE. Do not cut words.
2. **Helper words around a place are outside** unless they belong to the name: in `kadıköy civarında` the word
   `civarında` only says "around", so mark only [kadıköy]. `Avrupa yakasında` names a side of Istanbul, so both
   words are PLACE.
3. **DATE takes the whole time expression**, including `bu`, `gelecek`, `önümüzdeki`, clock times and `-den sonra`:
   [bu hafta sonu], [önümüzdeki cuma akşamı], [saat 9'dan sonra]. Two separate times are two spans.
4. **TYPE includes descriptive words that change the kind**: [açık hava konseri], [rock konseri], [çocuk
   tiyatrosu]. Generic words are not TYPE: `etkinlik`, `bir şey`, `aktivite`, `plan` stay unlabeled.
5. **NAME vs TYPE:** `Fenerbahçe maçı` → [Fenerbahçe] NAME, [maçı] TYPE. `Duman konseri` → [Duman] NAME,
   [konseri] TYPE.
6. **PRICE takes the full limit**: [en fazla 500 TL], [500 liradan ucuz], [1000 TL'yi geçmesin]. The verb that makes
   it a limit (`geçmesin`) is inside when it directly follows the amount.
7. **PEOPLE takes the count or the companion**: [iki kişi], [3 kişi], [annemle], [arkadaşlarımla]. A price per person
   (`kişi başı`) belongs to PRICE, not PEOPLE.
8. **EXCLUDE marks only what is excluded**, not the negating word: `konser olmasın` → [konser] EXCLUDE, `olmasın`
   unlabeled. A wanted kind in the same request is still TYPE: `tiyatro olsun konser olmasın` → [tiyatro] TYPE,
   [konser] EXCLUDE.
9. If nothing is asked for (`canım sıkıldı`), `spans` is empty.
