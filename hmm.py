"""Two count-based taggers written from scratch: the most-frequent-tag baseline and a bigram HMM decoded with Viterbi.

Words are lowercased and Turkish letters folded to ASCII (Kadıköy'de -> kadikoy'de), because users often type
without Turkish letters. Rare and unseen words are replaced by an unknown-word class (word_class) so the HMM can learn
emissions for them: numbers, words with an apostrophe suffix, and otherwise the last two letters, which in Turkish
are often a case ending (-de, -da, -te, -ta = "in/at").
"""

import math
from collections import Counter, defaultdict

from common import TAGS, bio_valid

FOLD = str.maketrans("çğıöşüâîûÇĞİÖŞÜÂÎÛ", "cgiosuaiuCGIOSUAIU")


def norm(word):
    return word.translate(FOLD).lower().replace("’", "'")


def word_class(word):
    """Unknown-word class of a normalized word."""
    if any(c.isdigit() for c in word):
        return "<NUM>"
    if "'" in word:
        return "<APOS>"
    if not word[0].isalnum():
        return "<PUNCT>"
    return f"<UNK-{word[-2:]}>"


class FrequentTag:
    """Baseline: each word gets the tag it had most often in training; unseen words get O."""

    def fit(self, rows):
        counts = defaultdict(Counter)
        for r in rows:
            for word, tag in zip(r["tokens"], r["tags"]):
                counts[norm(word)][tag] += 1
        self.best = {word: c.most_common(1)[0][0] for word, c in counts.items()}
        return self

    def predict(self, tokens):
        return [self.best.get(norm(w), "O") for w in tokens]


class HMM:
    """Bigram hidden Markov model: P(words, tags) = prod P(tag | previous tag) * P(word | tag).

    Training counts and normalizes with add-alpha smoothing. Words seen fewer than min_count times are replaced by
    their word_class, so the class emissions are learned from real rare words. Decoding is Viterbi in log space.
    With constrained=True, transitions that break IOB2 (O -> I-PLACE, B-DATE -> I-TYPE) get probability 0.
    """

    def __init__(self, alpha=0.1, min_count=2, constrained=False):
        self.alpha, self.min_count, self.constrained = alpha, min_count, constrained

    def symbol(self, word):
        word = norm(word)
        return word if self.vocab_counts[word] >= self.min_count else word_class(word)

    def fit(self, rows):
        self.vocab_counts = Counter(norm(w) for r in rows for w in r["tokens"])
        start, trans, emit = Counter(), defaultdict(Counter), defaultdict(Counter)
        for r in rows:
            previous = None
            for word, tag in zip(r["tokens"], r["tags"]):
                emit[tag][self.symbol(word)] += 1
                if previous is None:
                    start[tag] += 1
                else:
                    trans[previous][tag] += 1
                previous = tag
        symbols = {s for c in emit.values() for s in c}
        k, n = len(TAGS), len(symbols) + 1  # +1: one shared slot for symbols never seen with any tag
        self.log_start = {t: math.log((start[t] + self.alpha) / (sum(start.values()) + self.alpha * k)) for t in TAGS}
        self.log_trans = {p: {t: math.log((trans[p][t] + self.alpha) / (sum(trans[p].values()) + self.alpha * k))
                              for t in TAGS} for p in TAGS}
        self.emit, self.emit_total, self.n_symbols = emit, {t: sum(emit[t].values()) for t in TAGS}, n
        if self.constrained:
            for t in TAGS:
                if not bio_valid("O", t):
                    self.log_start[t] = -math.inf
                for p in TAGS:
                    if not bio_valid(p, t):
                        self.log_trans[p][t] = -math.inf
        return self

    def log_emit(self, tag, symbol):
        return math.log((self.emit[tag][symbol] + self.alpha) / (self.emit_total[tag] + self.alpha * self.n_symbols))

    def predict(self, tokens):
        """Viterbi: best[t] is the log probability of the best tag path for the words so far that ends in tag t."""
        if not tokens:
            return []
        symbols = [self.symbol(w) for w in tokens]
        best = {t: self.log_start[t] + self.log_emit(t, symbols[0]) for t in TAGS}
        back = []
        for symbol in symbols[1:]:
            step, pointers = {}, {}
            for t in TAGS:
                p = max(TAGS, key=lambda q: best[q] + self.log_trans[q][t])
                step[t] = best[p] + self.log_trans[p][t] + self.log_emit(t, symbol)
                pointers[t] = p
            best = step
            back.append(pointers)
        tag = max(TAGS, key=best.get)
        path = [tag]
        for pointers in reversed(back):
            tag = pointers[tag]
            path.append(tag)
        return path[::-1]
