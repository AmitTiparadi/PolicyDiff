"""Verify a downloaded Colab evidence bundle without loading model weights."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from scripts.finetune_translation import (
    DATASET, DATASET_REVISION, TRANSLATION_MODEL, TRANSLATION_REVISION,
    assert_disjoint, fingerprint, score_translations, write_json,
)


def verify(folder):
    def read(name):
        return json.loads((folder / name).read_text(encoding='utf-8'))
    config, manifest, result = read('run_config.json'), read('data_manifest.json'), read('comparison.json')
    assert result['status'] == 'completed'
    assert config['model'] == TRANSLATION_MODEL and config['model_revision'] == TRANSLATION_REVISION
    assert manifest['dataset'] == DATASET and manifest['revision'] == DATASET_REVISION
    splits = {}
    for name in ['train', 'validation', 'test']:
        splits[name] = [json.loads(line) for line in (folder / f'{name}.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(splits[name]) == manifest['splits'][name]['count'] == config[f'{name}_size']
        assert fingerprint(splits[name]) == manifest['splits'][name]['sha256']
    assert_disjoint(splits)
    assert result['test_sha256'] == manifest['splits']['test']['sha256']
    rows = list(csv.DictReader((folder / 'test_predictions.csv').open(encoding='utf-8-sig', newline='')))
    baseline = read('pretrained_predictions.json')
    assert len(rows) == len(baseline) == len(splits['test']) == result['test_count']
    def safe(value):
        return "'" + value if value.startswith(('=', '+', '-', '@')) else value
    def unsanitize(value):
        return value[1:] if len(value) > 1 and value[0] == "'" and value[1] in '=+-@' else value
    for row, pair, prediction in zip(rows, splits['test'], baseline):
        assert row['english'] == safe(pair['en'])
        assert row['reference_hindi'] == safe(pair['hi'])
        assert row['pretrained_hindi'] == safe(prediction)
    references = [r['hi'] for r in splits['test']]
    computed = {}
    for name, predictions in [('pretrained', baseline), ('fine_tuned', [unsanitize(r['fine_tuned_hindi']) for r in rows])]:
        computed[name] = score_translations(predictions, references)
        for metric in ['bleu', 'chrf']:
            assert math.isclose(computed[name][metric], result[name][metric], abs_tol=1e-8)
    assert math.isclose(result['bleu_delta'], computed['fine_tuned']['bleu'] - computed['pretrained']['bleu'], abs_tol=1e-8)
    history = read('loss_history.json')
    evaluations = [r for r in history if 'eval_loss' in r and r.get('epoch', 0) > 0]
    assert len(evaluations) == config['epochs']
    assert math.isclose(evaluations[-1]['epoch'], config['epochs'])
    assert all(math.isfinite(r['eval_loss']) for r in evaluations)
    best = min(evaluations, key=lambda r: r['eval_loss'])
    assert result['selected_checkpoint'].endswith(f'checkpoint-{best["step"]}')
    assert math.isclose(best['eval_loss'], result['best_validation_loss'], abs_tol=1e-6)
    assert math.isfinite(result['initial_validation_loss'])
    return {'status': 'verified', 'checks': [
        'Model and dataset revision pins', 'Sample counts and SHA-256 fingerprints',
        'No normalized English or Hindi overlap among supplied splits',
        'Prediction rows align with the same held-out references',
        'Recomputed corpus BLEU and chrF match recorded scores',
        'Recorded checkpoint corresponds to minimum epoch validation loss'],
        'recomputed_scores': computed, 'validation_epochs': evaluations,
        'selected_epoch': best['epoch'],
        'limits': 'Verifies exported evidence consistency; no model weights supplied, so model inference/training was not independently rerun. Initial validation loss is recorded in comparison.json but the Trainer resets its log history when training starts.',
        'source_files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(folder.iterdir()) if p.is_file() and p.name != 'verification.json'}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    verification = verify(args.folder)
    if args.output:
        write_json(args.output, verification)
    print(json.dumps(verification, indent=2))
