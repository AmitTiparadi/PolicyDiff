"""Integration verification with real cached models and sourced historical files."""
import hashlib
import json
import platform
import os
import importlib.metadata
from policydiff.analysis import default_config, add_hindi, compare
from policydiff.report import html_report, json_report
from policydiff.settings import ROOT
from policydiff.text import token_experiments, load_file


def main():
    output = ROOT / 'outputs'
    output.mkdir(exist_ok=True)
    old, new = load_file(ROOT / 'data/examples/old.txt'), load_file(ROOT / 'data/examples/new.txt')
    report = compare(old, new)
    assert report['counts']['modified'] >= 2
    assert any('Numbers changed' in r['signals'] for r in report['changes'])
    assert any('Negation changed' in r['signals'] for r in report['changes'])
    add_hindi(report)
    for row in report['changes']:
        if row['status'] != 'unchanged':
            assert any('\u0900' <= c <= '\u097f' for text in row['hindi'].values() for c in text)
    (output / 'demo.json').write_text(json_report(report), encoding='utf-8')
    (output / 'demo.html').write_text(html_report(report), encoding='utf-8')
    print('Synthetic semantic + Hindi:', report['counts'], flush=True)
    tokens = token_experiments('We retained records for 30 days. Contact us, please.')
    assert any(r['token'] == '30' for r in tokens)
    assert any(r['lemma'] == 'retain' for r in tokens)
    assert all(r['lemma'] != 'u' for r in tokens)
    historical = []
    for folder in sorted((ROOT / 'data/historical').glob('*')):
        if not folder.is_dir() or not (folder / 'provenance.json').exists():
            continue
        provenance = json.loads((folder / 'provenance.json').read_text(encoding='utf-8'))
        for side in ['old', 'new']:
            expected = provenance[side].get('text_sha256') or provenance[side]['sha256']
            assert hashlib.sha256((folder / f'{side}.md').read_bytes()).hexdigest() == expected
            if (folder / f'{side}.html').exists():
                assert hashlib.sha256((folder / f'{side}.html').read_bytes()).hexdigest() == provenance[side].get('stored_html_sha256', provenance[side]['raw_sha256'])
        before, after = load_file(folder / 'old.md'), load_file(folder / 'new.md')
        engines = {}
        for engine in ['textual', 'tfidf', 'semantic']:
            result = compare(before, after, default_config(engine))
            assert result['old_clauses'] == sum(r['old'] is not None for r in result['changes'])
            assert result['new_clauses'] == sum(r['new'] is not None for r in result['changes'])
            engines[engine] = result['counts']
            if engine == 'semantic':
                (output / f'{folder.name}.json').write_text(json_report(result), encoding='utf-8')
                (output / f'{folder.name}.html').write_text(html_report(result), encoding='utf-8')
        historical.append({'pair': folder.name, 'old_clauses': result['old_clauses'],
                           'new_clauses': result['new_clauses'], 'engines': engines})
        print(folder.name, engines, flush=True)
    assert len(historical) == 7
    record = json.dumps({'model_and_translation_smoke': 'passed',
                                                            'historical_pairs': historical,
                                                            'note': 'Successful integration checks; not accuracy metrics.'}, indent=2)
    (output / 'integration_check.json').write_text(record, encoding='utf-8')
    results = ROOT / 'submission/results'
    results.mkdir(parents=True, exist_ok=True)
    (results / 'integration_check.json').write_text(record, encoding='utf-8')
    from policydiff.models import embedding_model
    environment = {'system': platform.system(), 'machine': platform.machine(), 'processor': platform.processor(),
                   'logical_cores': os.cpu_count(), 'python': platform.python_version(), 'embedding_device': str(embedding_model().device),
                   'packages': {name: importlib.metadata.version(name) for name in ['torch', 'sentence-transformers', 'transformers', 'scikit-learn', 'nltk', 'sacrebleu', 'streamlit']}}
    (results / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    print('Integration verification complete.', flush=True)


if __name__ == '__main__':
    main()
