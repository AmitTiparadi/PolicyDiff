from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import re
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer

from .settings import EMBEDDING_MODEL, EMBEDDING_REVISION
from .text import normalize, segment


def default_config(engine="semantic"):
    """Load development-selected settings without inspecting held-out policies."""
    import json
    from .settings import ROOT
    path = ROOT / 'policydiff/calibration.json'
    if path.exists() and engine == 'semantic':
        return Config(**json.loads(path.read_text(encoding='utf-8'))['config'])
    return Config(engine=engine, use_critical_checks=engine == 'semantic')


@dataclass(frozen=True)
class Config:
    engine: str = "semantic"
    match_threshold: float = 0.55
    reword_threshold: float = 0.88
    ambiguity_margin: float = 0.04
    use_critical_checks: bool = True

    def __post_init__(self):
        if self.engine not in {"semantic", "tfidf", "textual"}:
            raise ValueError("Unknown comparison engine.")
        if not 0 <= self.match_threshold <= self.reword_threshold <= 1:
            raise ValueError("Thresholds must satisfy 0 <= match <= reword <= 1.")
        if not 0 <= self.ambiguity_margin <= 1:
            raise ValueError("Invalid ambiguity margin.")


def tfidf_vectors(texts):
    return TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b").fit_transform(texts)


def exact_pairs(old, new):
    available = {}
    for j, clause in enumerate(new):
        available.setdefault(normalize(clause.text).lower(), []).append(j)
    pairs = []
    for i, clause in enumerate(old):
        key = normalize(clause.text).lower()
        if available.get(key):
            pairs.append((i, available[key].pop(0)))
    return pairs


def similarity_matrix(old, new, engine):
    # Exact matches do not need model inference or quadratic textual scoring.
    exact = exact_pairs(old, new)
    used_old, used_new = {i for i, _ in exact}, {j for _, j in exact}
    oi = [i for i in range(len(old)) if i not in used_old]
    nj = [j for j in range(len(new)) if j not in used_new]
    matrix = np.zeros((len(old), len(new)))
    if not oi or not nj:
        return matrix
    a, b = [old[i] for i in oi], [new[j] for j in nj]
    texts = [c.text for c in a + b]
    if engine == "semantic":
        from .models import semantic_vectors
        vectors = semantic_vectors(texts)
        scores = np.clip(vectors[:len(a)] @ vectors[len(a):].T, 0, 1)
    elif engine == "tfidf":
        vectors = tfidf_vectors(texts)
        scores = (vectors[:len(a)] @ vectors[len(a):].T).toarray()
    else:
        scores = np.array([[SequenceMatcher(None, normalize(x.text).lower(), normalize(y.text).lower(),
                                           autojunk=False).ratio() for y in b] for x in a])
    matrix[np.ix_(oi, nj)] = scores
    return matrix


def critical_changes(old, new):
    a, b = normalize(old).lower(), normalize(new).lower()
    signals = []
    numbers = r"\b\d+(?:[.,]\d+)*\b"
    if Counter(re.findall(numbers, a)) != Counter(re.findall(numbers, b)):
        signals.append("Numbers changed")
    units = r"\b(?:days?|weeks?|months?|years?|hours?|minutes?|percent)\b|%"
    if Counter(re.findall(units, a)) != Counter(re.findall(units, b)):
        signals.append("Duration or quantity units changed")
    negation = r"\b(?:not|no|never|without|cannot|neither|nor)\b|n't\b"
    if Counter(re.findall(negation, a)) != Counter(re.findall(negation, b)):
        signals.append("Negation changed")
    modals = r"\b(?:may|must|shall|will|can|required|optional)\b"
    if Counter(re.findall(modals, a)) != Counter(re.findall(modals, b)):
        signals.append("Permission or obligation wording changed")
    categories = {
        "email": r"\bemail\b|e-mail",
        "location": r"\blocation\b|geolocation",
        "payment": r"\bpayment\b|credit card",
        "biometric": r"\bbiometric\w*\b",
        "advertising": r"\badvertis\w*\b",
        "third parties": r"\bthird[- ]part(?:y|ies)\b",
        "sale": r"\bsell\b|\bsold\b|\bsale\b",
        "encryption": r"\bencrypt\w*\b",
        "consent": r"\bconsent\b",
    }
    changed = [name for name, pattern in categories.items() if bool(re.search(pattern, a)) != bool(re.search(pattern, b))]
    if changed:
        signals.append("Privacy topic wording changed: " + ", ".join(changed))
    return signals


def _row(status, old=None, new=None, score=None, signals=None, reason=""):
    return {"status": status, "old": old.to_dict() if old else None,
            "new": new.to_dict() if new else None, "similarity": round(float(score), 4) if score is not None else None,
            "signals": signals or [], "reason": reason}


def align(old, new, matrix, config):
    rows, used_old, used_new = [], set(), set()
    # Multiset matching protects repeated exact clauses and moved clauses.
    for i, j in exact_pairs(old, new):
        rows.append(_row("unchanged", old[i], new[j], 1, reason="Same text after case and whitespace normalization"))
        used_old.add(i)
        used_new.add(j)
    oi = [i for i in range(len(old)) if i not in used_old]
    nj = [j for j in range(len(new)) if j not in used_new]
    if oi and nj:
        scores = matrix[np.ix_(oi, nj)]
        # Dummy nodes let either side remain unmatched. Global assignment prevents reuse.
        rewards = np.zeros((len(oi) + len(nj), len(oi) + len(nj)))
        context = np.array([[0.015 if normalize(old[i].section).lower() == normalize(new[j].section).lower()
                             and old[i].section != "Policy" else 0 for j in nj] for i in oi])
        rewards[:len(oi), :len(nj)] = scores - config.match_threshold + context
        rewards[:len(oi), :len(nj)][scores < config.match_threshold] = -1
        left, right = linear_sum_assignment(rewards, maximize=True)
        for x, y in zip(left, right):
            if x >= len(oi) or y >= len(nj) or scores[x, y] < config.match_threshold:
                continue
            i, j = oi[x], nj[y]
            score = scores[x, y]
            alternatives = list(np.delete(scores[x], y)) + list(np.delete(scores[:, y], x))
            margin = score - max(alternatives) if alternatives else 1.0
            signals = critical_changes(old[i].text, new[j].text) if config.use_critical_checks else []
            if margin < config.ambiguity_margin:
                status, reason = "review", "Ambiguous correspondence; possible repeated, split, or merged clause"
            elif signals:
                status, reason = "modified", "Explicit change signals detected in corresponding clauses"
            elif score >= config.reword_threshold:
                status, reason = "reworded", f"High {config.engine} similarity; likely rewording, verify meaning"
            else:
                status, reason = "modified", "Corresponding clauses have different wording; inspect the evidence"
            rows.append(_row(status, old[i], new[j], score, signals, reason))
            used_old.add(i)
            used_new.add(j)
    rows.extend(_row("removed", c, reason="No counterpart above the matching threshold")
                for i, c in enumerate(old) if i not in used_old)
    rows.extend(_row("added", new=c, reason="No counterpart above the matching threshold")
                for j, c in enumerate(new) if j not in used_new)
    return sorted(rows, key=lambda r: ((r["new"] or r["old"])["id"], r["status"]))


def readability(text):
    import textstat
    return {"flesch_reading_ease": round(textstat.flesch_reading_ease(text), 2),
            "flesch_kincaid_grade": round(textstat.flesch_kincaid_grade(text), 2),
            "words": textstat.lexicon_count(text, removepunct=True),
            "sentences": len(segment(text))}


def summarize(rows, limit=5):
    candidates = [r for r in rows if r["status"] in {"modified", "added", "removed", "review"}]
    if not candidates:
        return []
    texts = [(r["new"] or r["old"])["text"] for r in candidates]
    vectors = tfidf_vectors(texts)
    graph = (vectors @ vectors.T).toarray()
    np.fill_diagonal(graph, 0)
    sums = graph.sum(axis=1, keepdims=True)
    transition = np.divide(graph, sums, out=np.full_like(graph, 1 / len(graph)), where=sums != 0)
    ranks = np.full(len(graph), 1 / len(graph))
    for _ in range(60):
        next_ranks = 0.15 / len(graph) + 0.85 * transition.T @ ranks
        if np.max(np.abs(next_ranks - ranks)) < 1e-7:
            break
        ranks = next_ranks
    # Consequential short edits precede centrality so they cannot disappear in TextRank.
    order = sorted(range(len(candidates)), key=lambda i: (bool(candidates[i]["signals"]), ranks[i]), reverse=True)
    return [{"status": candidates[i]["status"], "section": (candidates[i]["new"] or candidates[i]["old"])["section"],
             "old": candidates[i]["old"]["text"] if candidates[i]["old"] else None,
             "new": candidates[i]["new"]["text"] if candidates[i]["new"] else None,
             "signals": candidates[i]["signals"]} for i in order[:limit]]


def compare(old_text, new_text, config=None):
    config = config or default_config()
    old, new = segment(old_text), segment(new_text)
    if not old or not new:
        raise ValueError("Both policies must contain at least one English clause, beyond headings.")
    matrix = similarity_matrix(old, new, config.engine)
    rows = align(old, new, matrix, config)
    counts = {status: sum(r["status"] == status for r in rows)
              for status in ["unchanged", "reworded", "modified", "added", "removed", "review"]}
    keywords = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b\w\w+\b")
    try:
        features = keywords.fit_transform([old_text, new_text])
        weights = np.asarray(features.sum(axis=0)).ravel()
        key_terms = keywords.get_feature_names_out()[np.argsort(weights)[-10:][::-1]].tolist()
    except ValueError:
        key_terms = []
    before, after = readability(old_text), readability(new_text)
    return {"schema_version": 1, "config": asdict(config),
            "embedding_model": EMBEDDING_MODEL if config.engine == "semantic" else None,
            "embedding_revision": EMBEDDING_REVISION if config.engine == "semantic" else None,
            "counts": counts, "old_clauses": len(old), "new_clauses": len(new),
            "changes": rows, "summary": summarize(rows), "keywords": key_terms,
            "readability": {"old": before, "new": after,
                            "delta": {k: round(after[k] - before[k], 2) for k in before}},
            "textual_diff_ratio": round(SequenceMatcher(None, old_text.split(), new_text.split(), autojunk=False).ratio(), 4),
            "textual_diff_unit": "word",
            "notes": ["Default semantic thresholds were selected on AI-reviewed Google development pairs; independent human validation remains pending.",
                      "Split/merge clauses may appear as unmatched clauses or review items.",
                      "Readability estimates describe prose difficulty, not policy quality."]}


def add_hindi(report):
    from .models import translate_hindi
    from .settings import TRANSLATION_MODEL, TRANSLATION_REVISION
    work = [(r, side) for r in report["changes"] if r["status"] != "unchanged"
            for side in ["old", "new"] if r[side]]
    # Commit translations only after all have succeeded, preventing partial reports.
    translations = translate_hindi([r[side]["text"] for r, side in work])
    for (row, side), value in zip(work, translations):
        row.setdefault("hindi", {})[side] = value
    report["translation_model"] = TRANSLATION_MODEL
    report["translation_revision"] = TRANSLATION_REVISION
    return report
