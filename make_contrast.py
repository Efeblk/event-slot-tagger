"""Build the contrast set: minimal pairs where one word decides whether a span is wanted or excluded.

Each pair has a "want" request and an "exclude" request that differ only in the negating word (olsun -> olmasın,
istiyorum -> istemiyorum, civarında -> dışında, var mı -> dışında ne var). The target span is the same words in both;
its label must flip from TYPE / PLACE / NAME to EXCLUDE. Requests are built from templates that follow
data/GUIDELINES.md, so this is a diagnostic of one phenomenon, not a test of natural requests.

Writes data/contrast/contrast.jsonl: one request per line with its pair id, phenomenon, version, BIO tags and the
target span [label, start, end] (word positions, end exclusive).

Usage: python make_contrast.py
"""

import random

from common import ROOT, to_bio, write_jsonl

PAIRS_PER_TEMPLATE = 20
DATES = ["bu akşam", "yarın", "bu hafta sonu", "cumartesi akşamı", "cuma", "pazar günü", "yarın akşam",
         "önümüzdeki hafta"]
PLACES = [("Kadıköy", "kadıköyde"), ("Beşiktaş", "beşiktaşta"), ("Ankara", "Ankara'da"), ("İzmir", "İzmir'de"),
          ("Moda", "Moda'da"), ("Şişli", "şişlide"), ("Bursa", "Bursa'da"), ("Eskişehir", "eskişehirde"),
          ("Üsküdar", "üsküdarda"), ("Taksim", "taksimde")]
TYPES = ["konser", "tiyatro", "stand-up", "opera", "bale", "müzikal", "caz", "sergi", "festival", "sinema"]
NAMES = ["Duman", "Tarkan", "Sezen Aksu", "Cem Yılmaz", "Mor ve Ötesi", "Fenerbahçe", "Galatasaray", "Hadise", "Manga",
         "Athena"]
PEOPLE = ["iki kişi", "arkadaşlarla", "sevgilimle", "3 kişiyiz"]

# (phenomenon, want template, exclude template, wanted label). {target} is the span whose label flips; the other
# slots keep their label in both versions.
TEMPLATES = [
    ("TYPE", "{date} {place} {target} olsun", "{date} {place} {target} olmasın", "TYPE"),
    ("TYPE", "{place} {target} istiyorum, {people}", "{place} {target} istemiyorum, {people}", "TYPE"),
    ("TYPE", "{date} {place} {target} var mı", "{date} {place} {target} dışında ne var", "TYPE"),
    ("PLACE", "{date} {type}, {target} olsun", "{date} {type}, {target} olmasın", "PLACE"),
    ("PLACE", "{date} {target} civarında {type}", "{date} {target} dışında {type}", "PLACE"),
    ("NAME", "{date} {target} olsun, {place}", "{date} {target} olmasın, {place}", "NAME"),
    ("NAME", "{date} {target} var mı", "{date} {target} hariç ne var", "NAME"),
]
SLOT_LABELS = {"date": "DATE", "place": "PLACE", "type": "TYPE", "people": "PEOPLE"}


def build(template, fill, target, target_label):
    """Text and BIO tags of one request, with the target span's word positions."""
    text = template.format(target=target, **fill)
    order = sorted((template.index("{" + slot + "}"), slot) for slot in ["target"] + list(fill)
                   if "{" + slot + "}" in template)
    span_list = [[target_label, target] if slot == "target" else [SLOT_LABELS[slot], fill[slot]] for _, slot in order]
    tokens, tags = to_bio(text, span_list)
    start = next(i for i, t in enumerate(tags) if t == f"B-{target_label}")
    end = start + 1
    while end < len(tags) and tags[end] == f"I-{target_label}":
        end += 1
    return text, tokens, tags, [target_label, start, end]


def main():
    rng = random.Random(0)
    rows, seen = [], set()
    for n, (phenomenon, want, exclude, label) in enumerate(TEMPLATES):
        made = 0
        while made < PAIRS_PER_TEMPLATE:
            fill = {"date": rng.choice(DATES), "place": rng.choice(PLACES)[1], "type": rng.choice(TYPES),
                    "people": rng.choice(PEOPLE)}
            target = {"TYPE": rng.choice(TYPES), "PLACE": rng.choice(PLACES)[1], "NAME": rng.choice(NAMES)}[label]
            if want.find("civarında") >= 0:
                target = rng.choice(PLACES)[0]  # bare place name before civarında / dışında
            fill = {k: v for k, v in fill.items() if "{" + k + "}" in want}
            if fill.get("type") == target:
                continue
            key = want.format(target=target, **fill)
            if key in seen:
                continue
            seen.add(key)
            pair = f"{phenomenon.lower()}-{n}-{made}"
            for version, template, target_label in (("want", want, label), ("exclude", exclude, "EXCLUDE")):
                text, tokens, tags, span = build(template, fill, target, target_label)
                rows.append({"id": f"contrast:{pair}:{version}", "pair": pair, "phenomenon": phenomenon,
                             "version": version, "text": text, "tokens": tokens, "tags": tags, "target": span})
            made += 1
    write_jsonl(ROOT / "data" / "contrast" / "contrast.jsonl", rows)
    print(f"wrote data/contrast/contrast.jsonl: {len(rows)} requests, {len(rows) // 2} pairs")


if __name__ == "__main__":
    main()
