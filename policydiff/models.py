"""Models load lazily and stay cached across comparisons."""
from functools import lru_cache
import numpy as np
from .settings import EMBEDDING_MODEL, EMBEDDING_REVISION, TRANSLATION_MODEL, TRANSLATION_REVISION


@lru_cache(maxsize=1)
def embedding_model(allow_download=False):
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import snapshot_download
    # A local snapshot path also prevents optional processor probes from contacting the Hub.
    path = EMBEDDING_MODEL if allow_download else snapshot_download(EMBEDDING_MODEL, revision=EMBEDDING_REVISION, local_files_only=True)
    return SentenceTransformer(path, revision=EMBEDDING_REVISION if allow_download else None, local_files_only=not allow_download)


def semantic_vectors(texts):
    model = embedding_model()
    tokenizer = model.tokenizer
    capacity = model.max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)
    chunks, owners = [], []
    def bounded(ids):
        text = tokenizer.decode(ids, skip_special_tokens=True)
        # Decoding a partial word can expand token count when it is encoded again.
        size = len(tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False))
        if size <= capacity:
            return [(text, len(ids))]
        if len(ids) <= 1:
            raise ValueError('A token exceeds the model input capacity.')
        mid = len(ids) // 2
        return bounded(ids[:mid]) + bounded(ids[mid:])
    for owner, text in enumerate(texts):
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
        for start in range(0, len(ids), capacity):
            for value, weight in bounded(ids[start:start + capacity]):
                chunks.append(value)
                owners.append((owner, weight))
    embeddings = model.encode(chunks, normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False, batch_size=32)
    vectors = np.zeros((len(texts), embeddings.shape[1]))
    for vector, (owner, weight) in zip(embeddings, owners):
        vectors[owner] += vector * weight
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return vectors


@lru_cache(maxsize=1)
def translation_model(allow_download=False):
    from transformers import MarianMTModel, MarianTokenizer
    from huggingface_hub import snapshot_download
    path = TRANSLATION_MODEL if allow_download else snapshot_download(TRANSLATION_MODEL, revision=TRANSLATION_REVISION, local_files_only=True)
    tokenizer = MarianTokenizer.from_pretrained(path, revision=TRANSLATION_REVISION if allow_download else None,
                                               local_files_only=not allow_download)
    model = MarianMTModel.from_pretrained(path, revision=TRANSLATION_REVISION if allow_download else None,
                                         local_files_only=not allow_download, use_safetensors=False)
    model.eval()
    return tokenizer, model


def translate_hindi(texts):
    """Translate complete clauses in bounded token chunks without dropping the tail."""
    import torch
    tokenizer, model = translation_model()
    results = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        parts = []
        for start in range(0, len(ids), 350):
            piece = tokenizer.decode(ids[start:start + 350], skip_special_tokens=True)
            inputs = tokenizer(piece, return_tensors="pt", truncation=False)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=512, num_beams=4)
            if output[0, -1].item() != tokenizer.eos_token_id:
                raise RuntimeError("Translation reached its output limit; compare a shorter clause.")
            parts.append(tokenizer.decode(output[0], skip_special_tokens=True))
        results.append(" ".join(parts))
    return results
