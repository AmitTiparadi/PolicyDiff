"""Synthetic diagnostics isolate consequential small edits; separate from historical metrics."""
from dataclasses import replace
import json
from policydiff.analysis import Config, align, similarity_matrix
from policydiff.settings import ROOT
from policydiff.text import segment

CASES = [
    ('retention', 'We retain your account records for 30 days after account deletion.', 'We retain your account records for 90 days after account deletion.', 'modified'),
    ('age', 'Our service is intended for users who are at least 18 years old.', 'Our service is intended for users who are at least 21 years old.', 'modified'),
    ('unit', 'We store your payment records for 30 days.', 'We store your payment records for 30 years.', 'modified'),
    ('negation', 'We do not share your personal information with advertisers.', 'We share your personal information with advertisers.', 'modified'),
    ('permission', 'We will delete your records when you close your account.', 'We may delete your records when you close your account.', 'modified'),
    ('data type', 'We collect your email address to process your account request.', 'We collect your location to process your account request.', 'modified'),
    ('paraphrase', 'You can ask us to delete your personal data.', 'You may request deletion of your personal information.', 'reworded'),
    ('punctuation', 'We protect your account records with encryption.', 'We protect your account records with encryption!', 'reworded'),
    ('unchanged', 'We retain records for 30 days.', 'We retain records for 30 days.', 'unchanged')
]


def main():
    chosen = Config(**json.loads((ROOT / 'policydiff/calibration.json').read_text())['config'])
    # Also show the handoff's .88 setting; neither diagnostic is used to tune thresholds.
    result = {'authority': 'Nine AI-authored synthetic stress cases; not historical evidence', 'thresholds': [], 'cases': []}
    for threshold in [.88, chosen.reword_threshold]:
        correct = {'without_checks': 0, 'with_checks': 0}
        for topic, old, new, label in CASES:
            a, b = segment(old), segment(new)
            matrix = similarity_matrix(a, b, 'semantic')
            row = {'topic': topic, 'old': old, 'new': new, 'reference': label, 'reword_threshold': threshold}
            for checks in [False, True]:
                key = 'with_checks' if checks else 'without_checks'
                rows = align(a, b, matrix, replace(chosen, reword_threshold=threshold, use_critical_checks=checks))
                guess = rows[0]['status'] if len(rows) == 1 else 'unmatched'
                row[key] = guess
                row['similarity'] = float(matrix[0, 0])
                correct[key] += guess == label
            result['cases'].append(row)
        result['thresholds'].append({'reword_threshold': threshold, 'count': len(CASES), 'correct': correct})
    (ROOT / 'submission/results/stress_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result['thresholds'])


if __name__ == '__main__':
    main()
