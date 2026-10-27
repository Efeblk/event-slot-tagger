"""Linear-chain CRF tagger (sklearn-crfsuite) with hand-made word features.

The CRF scores a whole tag sequence: feature weights for each word's tag plus a weight for every pair of neighbouring
tags, normalized over all sequences, trained by maximizing the log probability of the gold tags (L-BFGS with L1 and L2
penalties c1, c2). Decoding is Viterbi. The suffix features let it see Turkish case endings (kadıköy-de, harbiye-de).
"""

import sklearn_crfsuite

from hmm import norm


def word_features(tokens, i, suffixes=True):
    word = tokens[i]
    low = norm(word)
    stem, _, ending = low.partition("'")
    feats = {
        "bias": 1.0,
        "word": low,
        "stem": stem,
        "is_title": word[:1].isupper(),
        "has_digit": any(c.isdigit() for c in word),
        "is_punct": not word[:1].isalnum(),
        "has_turkish_letter": low != word.lower().replace("’", "'"),
        "length": min(len(word), 8),
    }
    if suffixes:
        feats["after_apostrophe"] = ending
        for n in (1, 2, 3, 4):
            feats[f"suffix{n}"] = low[-n:]
        feats["prefix3"] = low[:3]
    for offset in (-2, -1, 1, 2):
        j = i + offset
        feats[f"word{offset:+d}"] = norm(tokens[j]) if 0 <= j < len(tokens) else "<BOUNDARY>"
    for offset in (-1, 1):
        j = i + offset
        if suffixes and 0 <= j < len(tokens):
            feats[f"suffix2{offset:+d}"] = norm(tokens[j])[-2:]
    return feats


class CRF:
    def __init__(self, c1=0.05, c2=0.05, suffixes=True, max_iterations=200):
        self.suffixes = suffixes
        self.model = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=max_iterations,
                                          all_possible_transitions=True)

    def features(self, tokens):
        return [word_features(tokens, i, self.suffixes) for i in range(len(tokens))]

    def fit(self, rows):
        self.model.fit([self.features(r["tokens"]) for r in rows], [r["tags"] for r in rows])
        return self

    def predict(self, tokens):
        return list(self.model.predict_single(self.features(tokens))) if tokens else []

    def transitions(self):
        """Learned score of every (previous tag, tag) pair."""
        return {f"{a} -> {b}": w for (a, b), w in self.model.transition_features_.items()}
