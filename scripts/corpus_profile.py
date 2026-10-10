"""Profile a deterministic PrivaSeer sample; these are corpus statistics, not accuracy."""
import csv
import hashlib
import json
import re
from policydiff.settings import ROOT
import textstat
from policydiff.text import load_file, token_experiments


def main():
    base = ROOT / 'PrivaSeer-Corpus-main-demo/demo'
    files = sorted((base / 'text').glob('*.txt'))
    if not files:
        # The compact submission includes this deterministic sample.
        files = sorted((ROOT / 'data/corpus_sample').glob('*.txt'))
    if not files:
        raise FileNotFoundError('Source PrivaSeer demo or included data/corpus_sample is required.')
    selected, rejected = [], []
    for path in files:
        try:
            load_file(path)
        except ValueError as exc:
            rejected.append({'document': path.name, 'reason': str(exc)})
            continue
        selected.append(path)
        if len(selected) == 25:
            break
    out = ROOT / 'submission/results'
    out.mkdir(parents=True, exist_ok=True)
    sample = ROOT / 'data/corpus_sample'
    sample.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in selected:
        text = load_file(path)
        (sample / path.name).write_bytes(path.read_bytes()) if path.parent != sample else None
        rows.append({'document_id': path.stem, 'words': textstat.lexicon_count(text),
                     'numeric_tokens': len(re.findall(r'\b\d+\b', text)),
                     'flesch_reading_ease': textstat.flesch_reading_ease(text),
                     'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    with (out / 'corpus_profile.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    record = {'source': 'https://git.psu.edu/hlt-lab/PrivaSeer-Corpus', 'local_demo_txt_count': len(files),
              'local_demo_html_count': len(list((base / 'html').glob('*.html'))), 'profiled_count': len(selected),
              'selection': 'First 25 loadable English documents, sorted lexicographically; convenience sample',
              'rejected_before_selection_complete': rejected,
              'role': 'Preprocessing/readability demonstration. Single snapshots are not historical comparison pairs.',
              'sample_sha256': {r['document_id']: r['sha256'] for r in rows}}
    (sample / 'provenance.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    tokens = token_experiments('We retained records for 30 days. We do not share email. Contact us, please.')
    with (out / 'token_experiments.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tokens[0])); writer.writeheader(); writer.writerows(tokens)
    print(record['local_demo_txt_count'], 'available;', len(rows), 'profiled')


if __name__ == '__main__':
    main()
