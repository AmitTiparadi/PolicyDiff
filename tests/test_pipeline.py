import json
import numpy as np
import pytest
from policydiff.analysis import Config, align, compare, critical_changes, summarize
from policydiff.report import highlighted, html_report, json_report
from policydiff.text import load_bytes, segment


def test_loading_preserves_policy_header_and_removes_scripts():
    text = load_bytes(b'<header><h1>Privacy Policy</h1><p>We retain records for 30 days.</p></header><script>steal()</script>', 'policy.html')
    assert "30 days" in text and "steal" not in text
    assert segment(text)[0].section == "Privacy Policy"


def test_offsets_and_decimal_sentence_boundaries():
    text = 'Retention\nWe charge 2.5 percent. We retain data for 30 days.'
    clauses = segment(text)
    assert len(clauses) == 2
    for clause in clauses:
        assert text[clause.start:clause.end].strip() == clause.text


@pytest.mark.parametrize('old,new,signal', [
    ('We retain data for 30 days.', 'We retain data for 90 days.', 'Numbers changed'),
    ('We retain data for 30 days.', 'We retain data for 30 years.', 'Duration or quantity units changed'),
    ('We do not share data.', 'We share data.', 'Negation changed'),
    ("We don’t sell data.", 'We sell data.', 'Negation changed'),
    ('We may collect location.', 'We must collect location.', 'Permission or obligation wording changed'),
    ('We collect email.', 'We collect location.', 'Privacy topic wording changed: email, location'),
])
def test_critical_edits(old, new, signal):
    assert signal in critical_changes(old, new)


def test_high_similarity_cannot_hide_changed_number():
    old = segment('We retain data for 30 days.')
    new = segment('We retain data for 90 days.')
    result = align(old, new, np.array([[0.99]]), Config())
    assert result[0]['status'] == 'modified'
    assert 'Numbers changed' in result[0]['signals']


def test_global_assignment_never_reuses_a_clause():
    old = segment('We collect email. We collect your location.')
    new = segment('We collect your email address. We collect precise location.')
    result = align(old, new, np.array([[0.91, 0.70], [0.85, 0.89]]), Config())
    assert len({r['new']['id'] for r in result if r['new']}) == 2
    assert len({r['old']['id'] for r in result if r['old']}) == 2


def test_duplicate_exact_matches_and_reordering():
    old = segment('We collect email. We collect email. We encrypt data.')
    new = segment('We encrypt data. We collect email.')
    result = align(old, new, np.zeros((3, 2)), Config())
    assert sum(r['status'] == 'unchanged' for r in result) == 2
    assert sum(r['status'] == 'removed' for r in result) == 1


def test_unrelated_clauses_become_addition_and_removal():
    rows = align(segment('We collect email.'), segment('Contact our office.'), np.array([[0.1]]), Config())
    assert {r['status'] for r in rows} == {'added', 'removed'}


def test_ambiguous_matches_are_reviewed():
    rows = align(segment('We share account data.'), segment('We share email data. We share location data.'),
                 np.array([[0.8, 0.79]]), Config())
    assert any(r['status'] == 'review' for r in rows)


def test_shared_classification_threshold_for_fair_baselines():
    old, new = segment('You can request deletion.'), segment('You can ask us to delete your data.')
    assert align(old, new, np.array([[0.94]]), Config())[0]['status'] == 'reworded'
    assert align(old, new, np.array([[0.94]]), Config(engine='textual'))[0]['status'] == 'reworded'


def test_ablation_can_disable_guards_without_changing_alignment():
    old, new = segment('We retain data for 30 days.'), segment('We retain data for 90 days.')
    result = align(old, new, np.array([[0.99]]), Config(use_critical_checks=False))
    assert result[0]['status'] == 'reworded'
    assert result[0]['signals'] == []


def test_markdown_frontmatter_and_changelog_are_not_policy_clauses():
    text = load_bytes(b'---\ntitle: Policy\n---\n## Retention\nWe retain data for 30 days.\n### Change log\nChanged yesterday.', 'policy.md')
    assert len(segment(text)) == 1
    assert '30 days' in text and 'Changed yesterday' not in text


def test_compare_report_is_serializable_and_readability_uses_prose():
    report = compare('We retain data for 30 days.', 'We retain data for 90 days.', Config(engine='textual'))
    assert report['counts']['modified'] == 1
    assert json.loads(json_report(report))['readability']['old']['words'] > 0
    assert 'Numbers changed' in html_report(report)


def test_highlights_escape_uploaded_markup():
    a, b = highlighted('<script>alert(1)</script>', '<img src=x>')
    assert '<script>' not in a and '<img' not in b


def test_summary_keeps_consequential_edit_before_centrality():
    rows = [dict(status='modified', old={'text': '30 days', 'section': 'Retention'},
                 new={'text': '90 days', 'section': 'Retention'}, signals=['Numbers changed'])]
    rows += [dict(status='added', old=None, new={'text': 'We collect email addresses and account data.', 'section': 'Collection'}, signals=[])] * 6
    assert summarize(rows, limit=1)[0]['signals'] == ['Numbers changed']


def test_invalid_input_and_configuration():
    with pytest.raises(ValueError):
        load_bytes(b'', 'empty.txt')
    with pytest.raises(ValueError):
        Config(match_threshold=0.9, reword_threshold=0.5)


def test_translation_failure_does_not_mutate_report(monkeypatch):
    from policydiff.analysis import add_hindi
    import policydiff.models
    report = compare('We retain data for 30 days.', 'We retain data for 90 days.', Config(engine='textual'))
    def fail(texts):
        raise RuntimeError('Model unavailable')
    monkeypatch.setattr(policydiff.models, 'translate_hindi', fail)
    with pytest.raises(RuntimeError):
        add_hindi(report)
    assert 'hindi' not in report['changes'][0]


def test_exact_matches_need_no_embedding_model(monkeypatch):
    import policydiff.models
    def fail(texts):
        raise AssertionError('Exact matches should bypass model inference')
    monkeypatch.setattr(policydiff.models, 'semantic_vectors', fail)
    report = compare('We collect email. We encrypt records.', 'We encrypt records. We collect email.')
    assert report['counts']['unchanged'] == 2
