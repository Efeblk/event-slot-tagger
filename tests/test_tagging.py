"""Checks of the span conversion, the scoring, and the HMM / CRF math against the week 6 worked examples."""

import math

import pytest

from common import TAGS, span_scores, spans, to_bio, tokenize
from hmm import HMM, norm, word_class


def test_tokenize_keeps_suffixes_and_times():
    assert tokenize("yarın İstanbul’da stand-up, 20:00'den sonra!") == [
        "yarın", "İstanbul’da", "stand-up", ",", "20:00'den", "sonra", "!"]
    assert tokenize("19.30dan sonra 500 tl") == ["19.30dan", "sonra", "500", "tl"]


def test_to_bio_whole_words_in_order():
    tokens, tags = to_bio("tiyatro veya stand-up, konser olmasın",
                          [["TYPE", "tiyatro"], ["TYPE", "stand-up"], ["EXCLUDE", "konser"]])
    assert tags == ["B-TYPE", "O", "B-TYPE", "O", "B-EXCLUDE", "O"]
    _, tags = to_bio("bu akşam kadıköyde", [["DATE", "bu akşam"], ["PLACE", "kadıköyde"]])
    assert tags == ["B-DATE", "I-DATE", "B-PLACE"]


def test_to_bio_rejects_part_of_a_word():
    with pytest.raises(ValueError):
        to_bio("kadıköyde konser", [["PLACE", "kadıköy"]])


def test_spans_and_exact_scoring():
    gold = [["B-PLACE", "O", "B-DATE", "I-DATE"]]
    assert spans(gold[0]) == {("PLACE", 0, 1), ("DATE", 2, 4)}
    pred = [["B-PLACE", "O", "O", "B-DATE"]]  # the date is cut short: one FP and one FN
    scores = span_scores(gold, pred)
    assert scores["precision"] == 0.5 and scores["recall"] == 0.5 and scores["f1"] == 0.5
    assert spans(["O", "I-TYPE", "I-TYPE"]) == {("TYPE", 1, 3)}


def test_word_classes():
    assert norm("Kadıköy’de") == "kadikoy'de"
    assert word_class("kadikoy'de") == "<APOS>"
    assert word_class("500") == "<NUM>"
    assert word_class("harbiyede") == "<UNK-de>"


def slide_hmm():
    """The 'can fish' HMM of the week 6 slides with tags N and V."""
    hmm = HMM()
    log = math.log
    hmm.tags = ("N", "V")
    hmm.log_start = {"N": log(0.6), "V": log(0.4)}
    hmm.log_trans = {"N": {"N": log(0.3), "V": log(0.7)}, "V": {"N": log(0.8), "V": log(0.2)}}
    emit = {"N": {"can": 0.4, "fish": 0.6}, "V": {"can": 0.7, "fish": 0.3}}
    return hmm, emit


def test_viterbi_matches_slide_example(monkeypatch):
    import hmm as hmm_module
    model, emit = slide_hmm()
    monkeypatch.setattr(hmm_module, "TAGS", model.tags)
    model.symbol = lambda word: word
    model.log_emit = lambda tag, symbol: math.log(emit[tag][symbol])
    assert model.predict(["can", "fish"]) == ["V", "N"]  # joint weight 0.1344, the largest of the four paths


def test_crf_layer_matches_slide_example():
    torch = pytest.importorskip("torch")
    from train_bert import viterbi
    ln2, ln8 = math.log(2), math.log(8)
    emissions = torch.tensor([[ln2, 0.0], [0.0, ln2]])  # can: N=ln2, V=0; fish: N=0, V=ln2
    transitions = torch.tensor([[0.0, 0.0], [ln8, 0.0]])  # a(V, N) = ln 8
    assert viterbi(emissions, transitions, torch.zeros(2)) == [1, 0]  # V N
    alpha = emissions[0]
    alpha = torch.logsumexp(alpha.unsqueeze(1) + transitions, dim=0) + emissions[1]
    assert torch.logsumexp(alpha, 0).exp().item() == pytest.approx(16.0)  # Z = 2 + 4 + 8 + 2

    from torch import nn
    from train_bert import Tagger
    layer = Tagger.__new__(Tagger)
    nn.Module.__init__(layer)
    layer.transitions = nn.Parameter(transitions)
    layer.start = nn.Parameter(torch.zeros(2))
    layer.end = nn.Parameter(torch.zeros(2))
    loss = layer.crf_nll(emissions.unsqueeze(0), torch.tensor([[1, 0]]), torch.tensor([[True, True]]))
    assert loss.item() == pytest.approx(math.log(2))  # -ln P(VN) = -ln(8/16)


def test_tag_set_size():
    assert len(TAGS) == 2 * 7 + 1


def test_perturbations_keep_words_aligned():
    from robustness import PERTURBATIONS, tr_lower, tr_upper
    tokens = ["İzmir'de", "Işık", "ücretsiz", "stand-up", "'", "20:00'den"]
    assert tr_lower("İzmir'de Işık") == "izmir'de ışık" and tr_upper("izmir ışık") == "İZMİR IŞIK"
    assert PERTURBATIONS["no_turkish_letters"](tokens, "k")[:3] == ["Izmir'de", "Isik", "ucretsiz"]
    assert PERTURBATIONS["no_apostrophes"](tokens, "k")[4] == "'"  # a lone apostrophe stays a token
    for name, change in PERTURBATIONS.items():
        changed = change(tokens, "k")
        assert len(changed) == len(tokens) and all(changed), name
        assert changed == change(tokens, "k"), name  # the same key gives the same typos


def test_contrast_pairs_flip_only_the_target():
    from common import ROOT, read_jsonl
    rows = read_jsonl(ROOT / "data" / "contrast" / "contrast.jsonl")
    pairs = {}
    for r in rows:
        pairs.setdefault(r["pair"], {})[r["version"]] = r
    assert len(pairs) == len(rows) // 2
    for pair in pairs.values():
        want, exclude = pair["want"], pair["exclude"]
        assert exclude["target"][0] == "EXCLUDE" and want["target"][0] in ("TYPE", "PLACE", "NAME")
        target_words = lambda r: r["tokens"][r["target"][1]:r["target"][2]]
        assert target_words(want) == target_words(exclude)
        others = lambda r: {(label, " ".join(r["tokens"][s:e])) for label, s, e in spans(r["tags"])} - {
            (r["target"][0], " ".join(target_words(r)))}
        assert others(want) == others(exclude)
