import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
os.environ.setdefault("HF_HOME", str(CACHE / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("NLTK_DATA", str(CACHE / "nltk"))
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-hi"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
TRANSLATION_REVISION = "75d7f7c9232b2891c7d65fe4ef635616c72be867"
