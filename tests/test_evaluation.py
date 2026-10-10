from scripts.benchmark import metrics
import pytest


def test_spurious_alignment_is_penalized_even_when_status_matches():
    gold = {(0, 0): 'modified', (1, None): 'removed'}
    predicted = {(0, 1): 'modified', (1, None): 'removed'}
    result = metrics(gold, predicted)
    assert result['alignment_precision'] == .5
    assert result['alignment_recall'] == .5
    assert result['change_f1'] == .5
    assert result['scored_pair_union'] == 3
    assert result['macro_f1_active'] < 1


def test_label_error_is_distinct_from_alignment_error():
    result = metrics({(0, 0): 'modified'}, {(0, 0): 'reworded'})
    assert result['alignment_precision'] == 1
    assert result['classification']['modified']['recall'] == 0
    assert result['macro_f1_active'] == 0


def test_human_mode_rejects_incomplete_review(tmp_path, monkeypatch):
    from scripts import benchmark
    (tmp_path / 'publisher.csv').write_text('old_id,new_id,label,authority,human_label\n0,0,modified,AI source review,\n', encoding='utf-8')
    monkeypatch.setattr(benchmark, 'BENCH', tmp_path)
    monkeypatch.setattr(benchmark, 'USE_HUMAN', True)
    with pytest.raises(ValueError, match='Independent review incomplete'):
        benchmark.gold_for(tmp_path / 'publisher')
