"""Prepare auditable source-linked annotations and run reproducible experiments."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from difflib import SequenceMatcher
from time import perf_counter
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from policydiff.analysis import Config, align, exact_pairs, similarity_matrix
from policydiff.settings import ROOT
from policydiff.text import load_file, segment

BENCH = ROOT / 'data' / 'benchmark'
LABELS = ['unchanged', 'reworded', 'modified', 'added', 'removed', 'review', 'no_pair']
USE_HUMAN = False


def pairs():
    return [f for f in sorted((ROOT / 'data/historical').iterdir()) if f.is_dir() and (f / 'provenance.json').exists()]


def prepare():
    result = []
    for folder in pairs():
        old, new = segment(load_file(folder / 'old.md')), segment(load_file(folder / 'new.md'))
        exact = exact_pairs(old, new)
        oi, nj = {i for i, _ in exact}, {j for _, j in exact}
        remaining_new = [c for c in new if c.id not in nj]
        residual = []
        for clause in old:
            if clause.id in oi:
                continue
            candidates = sorted(remaining_new, key=lambda b: SequenceMatcher(None, clause.text.lower(), b.text.lower(), autojunk=False).ratio(), reverse=True)
            residual.append({'old': clause.to_dict(), 'candidates': [b.to_dict() for b in candidates[:3]]})
        result.append({'pair': folder.name, 'old_total': len(old), 'new_total': len(new),
                       'exact_count': len(exact), 'old_residual': residual,
                       'new_residual': [b.to_dict() for b in remaining_new]})
    (ROOT / 'outputs').mkdir(exist_ok=True)
    (ROOT / 'outputs/review_candidates.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print([(r['pair'], r['old_total'], r['new_total'], len(r['old_residual']), len(r['new_residual'])) for r in result])


def materialize():
    decisions = json.loads((BENCH / 'reviewed_decisions.json').read_text(encoding='utf-8'))
    manifest = {'annotation_authority': 'Human labels requested; reviewer identity recorded separately' if USE_HUMAN else 'AI-reviewed reference labels; no independent human validation',
                'unchanged_authority': 'Exact normalized text anchors',
                'split': 'All Google pairs development; Mozilla and Automattic held out by publisher', 'pairs': []}
    for folder in pairs():
        old, new = segment(load_file(folder / 'old.md')), segment(load_file(folder / 'new.md'))
        review = decisions[folder.name]
        groups = review.get('excluded_groups', [])
        excluded_old = {i for g in groups for i in g['old_ids']}
        excluded_new = {j for g in groups for j in g['new_ids']}
        rows = [{'old_id': i, 'new_id': j, 'label': 'unchanged', 'rationale': 'Exact normalized text',
                 'authority': 'exact_text'} for i, j in exact_pairs(old, new)
                 if i not in excluded_old and j not in excluded_new]
        for i, j, label, rationale in review['decisions']:
            rows.append({'old_id': i, 'new_id': j, 'label': label, 'rationale': rationale, 'authority': 'AI source review'})
        used_old = [r['old_id'] for r in rows if r['old_id'] is not None]
        used_new = [r['new_id'] for r in rows if r['new_id'] is not None]
        if len(used_old) != len(set(used_old)) or len(used_new) != len(set(used_new)):
            raise ValueError(f'Duplicate reference correspondence in {folder.name}')
        # Explicit residual unmatched labels are only allowed when all source review is complete.
        if set(used_old) != set(range(len(old))) - excluded_old or set(used_new) != set(range(len(new))) - excluded_new:
            raise ValueError(f'Incomplete review in {folder.name}; missing old {set(range(len(old))) - set(used_old)}, new {set(range(len(new))) - set(used_new)}')
        path = BENCH / f'{folder.name}.csv'
        reviewed = {}
        if path.exists():
            with path.open(encoding='utf-8-sig', newline='') as stream:
                reviewed = {(r['old_id'], r['new_id']): r.get('human_label', '') for r in csv.DictReader(stream)}
        with path.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['old_id', 'new_id', 'old_text', 'new_text', 'label', 'rationale', 'authority', 'human_label'])
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, 'old_text': old[row['old_id']].text if row['old_id'] is not None else '',
                                 'new_text': new[row['new_id']].text if row['new_id'] is not None else '',
                                 'human_label': reviewed.get((str(row['old_id']) if row['old_id'] is not None else '',
                                                              str(row['new_id']) if row['new_id'] is not None else ''), '')})
        manifest['pairs'].append({'id': folder.name, 'publisher': folder.name.split('-')[0],
                                  'partition': 'development' if folder.name.startswith('google-') else 'test',
                                  'old_clauses': len(old), 'new_clauses': len(new), 'labels': dict(Counter(r['label'] for r in rows)),
                                  'excluded_groups': groups, 'excluded_old': sorted(excluded_old), 'excluded_new': sorted(excluded_new),
                                  'old_sha256': hashlib.sha256((folder / 'old.md').read_bytes()).hexdigest(),
                                  'new_sha256': hashlib.sha256((folder / 'new.md').read_bytes()).hexdigest(),
                                  'annotation_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (BENCH / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def gold_for(folder):
    with (BENCH / f'{folder.name}.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    labels = []
    for r in rows:
        if USE_HUMAN and r['authority'] != 'exact_text' and r.get('human_label', '') not in LABELS[:-1]:
            raise ValueError(f"Independent review incomplete: {folder.name} {r['old_id']} -> {r['new_id']}")
        labels.append(r['human_label'] if USE_HUMAN and r.get('human_label') else r['label'])
    return {(int(r['old_id']) if r['old_id'] else None, int(r['new_id']) if r['new_id'] else None): label for r, label in zip(rows, labels)}


def prediction(rows):
    return {(r['old']['id'] if r['old'] else None, r['new']['id'] if r['new'] else None): r['status'] for r in rows}


def scoped_prediction(folder, rows):
    review = json.loads((BENCH / 'reviewed_decisions.json').read_text(encoding='utf-8'))[folder.name]
    eo = {i for g in review.get('excluded_groups', []) for i in g['old_ids']}
    en = {i for g in review.get('excluded_groups', []) for i in g['new_ids']}
    return {k: v for k, v in prediction(rows).items() if k[0] not in eo and k[1] not in en}


def metrics(gold, predicted):
    union = sorted(set(gold) | set(predicted), key=str)
    truth, guesses = [gold.get(k, 'no_pair') for k in union], [predicted.get(k, 'no_pair') for k in union]
    shared = len(set(gold) & set(predicted))
    changed_gold = {k: v for k, v in gold.items() if v != 'unchanged'}
    changed_pred = {k: v for k, v in predicted.items() if v != 'unchanged'}
    true_changes = len(set(changed_gold) & set(changed_pred))
    classification = classification_report(truth, guesses, labels=LABELS, output_dict=True, zero_division=0)
    active = sorted(set(truth) | set(guesses))
    active_macro = classification_report(truth, guesses, labels=active, output_dict=True, zero_division=0)['macro avg']['f1-score']
    cp, cr = true_changes / max(1, len(changed_pred)), true_changes / max(1, len(changed_gold))
    return {'alignment_precision': shared / max(1, len(predicted)), 'alignment_recall': shared / max(1, len(gold)),
            'change_precision': cp, 'change_recall': cr, 'change_f1': 2 * cp * cr / (cp + cr) if cp + cr else 0,
            'macro_f1_active': active_macro, 'classification': classification,
            'confusion_matrix': confusion_matrix(truth, guesses, labels=LABELS).tolist(),
            'labels': LABELS, 'scored_pair_union': len(union)}


def experiments():
    materialize()
    manifest = json.loads((BENCH / 'manifest.json').read_text())
    cache, cold = {}, {}
    for folder in pairs():
        a, b = segment(load_file(folder / 'old.md')), segment(load_file(folder / 'new.md'))
        for engine in ['textual', 'tfidf', 'semantic']:
            start = perf_counter()
            matrix = similarity_matrix(a, b, engine)
            elapsed = perf_counter() - start
            cache[(folder.name, engine)] = (a, b, matrix)
            cold[(folder.name, engine)] = elapsed
    # Tune only on Google development pairs; each method gets the same search budget.
    variants = [('difflib', 'textual', False), ('TF-IDF', 'tfidf', False),
                ('SBERT alone', 'semantic', False), ('PolicyDiff', 'semantic', True)]
    results, tuning = {}, []
    for name, engine, checks in variants:
        best = None
        for threshold in [.4, .5, .55, .6, .7]:
            for reword in [.8, .88, .94]:
                config = Config(engine, threshold, reword, .04, checks)
                gold, predicted = {}, {}
                for folder in pairs():
                    if not folder.name.startswith('google-'):
                        continue
                    a, b, matrix = cache[(folder.name, engine)]
                    gold.update({(folder.name, *k): v for k, v in gold_for(folder).items()})
                    predicted.update({(folder.name, *k): v for k, v in scoped_prediction(folder, align(a, b, matrix, config)).items()})
                score = metrics(gold, predicted)['macro_f1_active']
                tuning.append({'method': name, 'match': threshold, 'reword': reword, 'development_macro_f1': score})
                candidate = (score, -abs(threshold - .55), -abs(reword - .88))
                if best is None or candidate > best[0]:
                    best = (candidate, config)
        config = best[1]
        by_pair, aggregate = {}, {}
        for partition in ['development', 'test']:
            gold, predicted = {}, {}
            for folder in pairs():
                if folder.name.startswith('google-') != (partition == 'development'):
                    continue
                a, b, matrix = cache[(folder.name, engine)]
                rows = align(a, b, matrix, config)
                actual, predicted_pair = gold_for(folder), scoped_prediction(folder, rows)
                by_pair[folder.name] = metrics(actual, predicted_pair)
                gold.update({(folder.name, *k): v for k, v in actual.items()})
                predicted.update({(folder.name, *k): v for k, v in predicted_pair.items()})
            aggregate[partition] = metrics(gold, predicted)
        # Warm end-to-end core runtime excludes model load, readability, UI and translation.
        timings = []
        for folder in pairs():
            if folder.name.startswith('google-'):
                continue
            a, b, _ = cache[(folder.name, engine)]
            repeats = []
            for _ in range(3):
                start = perf_counter()
                align(a, b, similarity_matrix(a, b, engine), config)
                repeats.append(perf_counter() - start)
            timings.append({'pair': folder.name, 'median_seconds': float(np.median(repeats)), 'repeats': repeats})
        results[name] = {'config': config.__dict__, 'aggregate': aggregate, 'pairs': by_pair,
                         'runtime': timings, 'mean_warm_core_seconds': float(np.mean([r['median_seconds'] for r in timings]))}
    # A second ablation holds PolicyDiff's selected thresholds fixed for causal comparison.
    cfg = Config(**results['PolicyDiff']['config'])
    controlled = {}
    for checks in [False, True]:
        gold, predicted = {}, {}
        for folder in pairs():
            if folder.name.startswith('google-'):
                continue
            a, b, matrix = cache[(folder.name, 'semantic')]
            gold.update({(folder.name, *k): v for k, v in gold_for(folder).items()})
            predicted.update({(folder.name, *k): v for k, v in scoped_prediction(folder, align(a, b, matrix, Config(cfg.engine, cfg.match_threshold, cfg.reword_threshold, cfg.ambiguity_margin, checks))).items()})
        controlled['with_checks' if checks else 'without_checks'] = metrics(gold, predicted)
    out = ROOT / 'submission/results'
    out.mkdir(parents=True, exist_ok=True)
    record = {'annotation_authority': manifest['annotation_authority'], 'manifest': manifest,
              'methods': results, 'controlled_ablation': controlled, 'threshold_search': tuning,
              'initial_scoring_seconds': {f'{k[0]}:{k[1]}': v for k, v in cold.items()},
              'metric_notes': ['Macro F1 includes active labels only, including no_pair for missed/spurious alignments.',
                               'Change precision/recall measures correspondence for non-unchanged rows, not semantic label correctness.',
                               'Model warm runtime is measured sequentially; no statistical significance claim.']}
    errors = []
    for folder in pairs():
        a, b, matrix = cache[(folder.name, 'semantic')]
        actual = gold_for(folder)
        predicted = scoped_prediction(folder, align(a, b, matrix, cfg))
        for i, j in sorted(set(actual) | set(predicted), key=str):
            expected, guess = actual.get((i, j), 'no_pair'), predicted.get((i, j), 'no_pair')
            if expected != guess:
                errors.append({'pair': folder.name, 'old_id': i, 'new_id': j, 'expected': expected,
                               'predicted': guess, 'old_text': a[i].text if i is not None else '',
                               'new_text': b[j].text if j is not None else ''})
    (out / 'error_cases.json').write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'benchmark_results.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    with (out / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'alignment_precision', 'alignment_recall', 'change_precision', 'change_recall', 'change_f1', 'macro_f1_active', 'warm_core_seconds'])
        for name, r in results.items():
            m = r['aggregate']['test']
            writer.writerow([name, *[m[k] for k in ['alignment_precision', 'alignment_recall', 'change_precision', 'change_recall', 'change_f1', 'macro_f1_active']], r['mean_warm_core_seconds']])
    # Save the selected configuration independently; app adopts it only after this run finishes.
    (ROOT / 'policydiff/calibration.json').write_text(json.dumps({'config': results['PolicyDiff']['config'],
        'basis': 'Google development pairs, AI-reviewed reference labels; publishers held out for evaluation'}, indent=2), encoding='utf-8')
    print(json.dumps({name: r['aggregate']['test']['macro_f1_active'] for name, r in results.items()}, indent=2), flush=True)


def main():
    global USE_HUMAN
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-review', action='store_true')
    parser.add_argument('--human-reviewed', action='store_true', help='Require a human label for every non-exact reference row; incomplete review is rejected.')
    args = parser.parse_args()
    USE_HUMAN = args.human_reviewed
    prepare() if args.prepare_review else experiments()


if __name__ == '__main__':
    main()
