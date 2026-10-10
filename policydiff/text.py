"""Load readable text; retain evidence and sentence locations."""
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import unicodedata

from bs4 import BeautifulSoup
from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer

MAX_BYTES = 5 * 1024 * 1024
MAX_CLAUSES = 600


@dataclass
class Clause:
    id: int
    text: str
    section: str
    start: int
    end: int

    def to_dict(self):
        return asdict(self)


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).translate(
        str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "\u00a0": " "})
    ).split())


def load_bytes(data, filename):
    if len(data) > MAX_BYTES:
        raise ValueError("Each policy must be at most 5 MB.")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".txt", ".html", ".htm", ".md"}:
        raise ValueError("Use TXT, HTML, or Markdown policy files.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("cp1252", errors="replace")
    if suffix in {".html", ".htm"}:
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style", "nav", "noscript"]):
            tag.decompose()
        # Keep policy headings and lists; avoid deleting headers that contain clauses.
        for tag in soup.find_all(["p", "li", "div", "h1", "h2", "h3", "h4", "br"]):
            tag.insert_before("\n## " if tag.name.startswith("h") and tag.name != "html" else "\n")
            tag.insert_after("\n")
        text = soup.get_text(separator=" ")
    elif suffix == ".md":
        # Revision-history entries describe old edits, rather than current obligations.
        text = re.split(r'(?im)^#{1,6}[ \t]+change[ \t]*log[ \t]*$', text)[0]
        text = re.sub(r'\A\s*(---|\+\+\+)\s*\n.*?\n\1\s*\n', '', text, flags=re.S)
        text = re.sub(r'(?m)^\s*\{:\s*[^\n]*\}\s*$', '', text)
        text = re.sub(r'(?m)^\s*[-*+]\s+', '', text)
        text = re.sub(r'\\([\[\]*_])', r'\1', text)
        text = re.sub(r'(?m)^Effective [^\n]*(?:Archived versions|Download PDF)[^\n]*$', '', text)
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        text = text.replace("**", "").replace("__", "")
    text = "\n".join(normalize(line) for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text or not re.search(r"[A-Za-z]", text):
        raise ValueError("No English policy text was found in this file.")
    return text


def load_file(path):
    path = Path(path)
    return load_bytes(path.read_bytes(), path.name)


def segment(text):
    params = PunktParameters()
    params.abbrev_types = {"e.g", "i.e", "etc", "mr", "mrs", "dr", "inc", "ltd", "u.s", "no"}
    tokenizer = PunktSentenceTokenizer(params)
    clauses, section = [], "Policy"
    for match in re.finditer(r"[^\n]+", text):
        line = match.group().strip()
        if not line:
            continue
        # Only clear headings are excluded. Short statements remain analyzable clauses.
        words = line.split()
        if re.match(r"^#{1,6}\s+", line):
            section = re.sub(r"^#{1,6}\s+", "", line).rstrip(":")
            continue
        if len(words) <= 9 and not re.search(r"[.!?]$", line) and (
            line.endswith(":") or line.isupper() or line.istitle()
        ):
            section = line.rstrip(":")
            continue
        for start, end in tokenizer.span_tokenize(match.group()):
            value = match.group()[start:end].strip()
            if re.search(r"[A-Za-z]", value):
                leading = len(match.group()[start:end]) - len(match.group()[start:end].lstrip())
                clauses.append(Clause(len(clauses), value, section, match.start() + start + leading,
                                      match.start() + end))
    if len(clauses) > MAX_CLAUSES:
        raise ValueError(f"This policy has over {MAX_CLAUSES} clauses. Compare individual sections.")
    return clauses


def token_experiments(text):
    """POS, stemming and lemmatization experiment, separate from semantic input."""
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tokenize import TreebankWordTokenizer
    from .settings import CACHE

    nltk.data.path.insert(0, str(CACHE / "nltk"))
    stemmer, lemmatizer = PorterStemmer(), WordNetLemmatizer()
    stop = set(stopwords.words("english")) - {"no", "not", "nor", "never"}
    rows = []
    for clause in segment(text):
        tokens = TreebankWordTokenizer().tokenize(normalize(clause.text))
        # Tag sentence tokens with case/punctuation intact, then filter and normalize.
        tagged = nltk.pos_tag(tokens)
        for token, tag in tagged:
            word = token.lower()
            if not any(c.isalnum() for c in word) or word in stop:
                continue
            pos = {"J": "a", "V": "v", "N": "n", "R": "r"}.get(tag[0])
            lemma = lemmatizer.lemmatize(word, pos) if pos else word
            rows.append({"clause_id": clause.id, "token": word, "pos": tag,
                         "stem": stemmer.stem(word), "lemma": lemma})
    return rows
