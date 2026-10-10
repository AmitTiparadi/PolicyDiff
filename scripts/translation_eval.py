"""Evaluate pinned Hindi MT against aligned, publisher-written Hindi references."""
import csv
import hashlib
import json
from time import perf_counter
from bs4 import BeautifulSoup
from sacrebleu.metrics import BLEU, CHRF
from policydiff.models import translate_hindi
from policydiff.settings import ROOT, TRANSLATION_MODEL, TRANSLATION_REVISION
from scripts.source_data import sanitize_html

URL = 'https://policies.google.com/privacy/archive/20240328?hl=hi'
# Paragraph correspondences inspected in both published pages, not model-selected.
INDICES = [0, 1, 12, 14, 15, 20, 26, 28, 29, 31, 36, 37, 40, 41, 45]


def main():
    english_path = ROOT / 'data/historical/google-20240328-20240916/old.html'
    hindi_path = ROOT / 'data/benchmark/google_hindi_reference.html'
    if not hindi_path.exists():
        import requests
        response = requests.get(URL, timeout=60)
        response.raise_for_status()
        hindi_path.write_bytes(sanitize_html(response.content))
    def paragraphs(path):
        soup = BeautifulSoup(path.read_bytes(), 'html.parser')
        return [p.get_text(' ', strip=True) for p in (soup.find('main') or soup).find_all('p')]
    en, hi = paragraphs(english_path), paragraphs(hindi_path)
    sources, refs = [en[i] for i in INDICES], [hi[i] for i in INDICES]
    assert all(any('\u0900' <= c <= '\u097f' for c in t) for t in refs)
    start = perf_counter()
    outputs = translate_hindi(sources)
    elapsed = perf_counter() - start
    bleu, chrf = BLEU(tokenize='intl'), CHRF()
    sentence_bleu = BLEU(tokenize='intl', effective_order=True)
    b, c = bleu.corpus_score(outputs, [refs]), chrf.corpus_score(outputs, [refs])
    result = {'model': TRANSLATION_MODEL, 'revision': TRANSLATION_REVISION,
              'reference_source': URL, 'english_source': URL.replace('?hl=hi', '?hl=en'),
              'reference_authority': 'Publisher-written Hindi; paragraph alignment reviewed by AI, not independently by a bilingual reviewer',
              'paragraph_indices': INDICES, 'paragraph_count': len(INDICES),
              'bleu': b.score, 'chrf': c.score, 'bleu_signature': str(bleu.get_signature()),
              'chrf_signature': str(chrf.get_signature()), 'total_seconds_including_model_load': elapsed,
              'source_sha256': hashlib.sha256(hindi_path.read_bytes()).hexdigest(),
              'notes': ['Single-publisher, 15-paragraph diagnostic; not a representative Hindi benchmark.',
                        'BLEU and chrF measure reference overlap, not legal adequacy.',
                        'No fine-tuning or claim of bilingual human validation.']}
    out = ROOT / 'submission/results'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'translation_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    with (out / 'translation_examples.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['paragraph_index', 'english', 'published_hindi_reference', 'model_hindi', 'sentence_bleu', 'sentence_chrf', 'human_adequacy_1_to_5', 'human_notes'])
        for i, a, reference, output in zip(INDICES, sources, refs, outputs):
            writer.writerow([i, a, reference, output, sentence_bleu.sentence_score(output, [reference]).score,
                             chrf.sentence_score(output, [reference]).score, '', ''])
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
