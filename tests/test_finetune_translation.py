import pytest

from scripts.finetune_translation import assert_disjoint, fingerprint, normalize, select_rows


class Tokenizer:
    def __call__(self, text, text_target, truncation):
        assert truncation is False
        return {'input_ids': text.split(), 'labels': text_target.split()}


def record(en, hi):
    return {'translation': {'en': en, 'hi': hi}}


def test_sampling_rejects_overlap_duplicates_empty_and_long_pairs():
    rows, counts = select_rows([
        record('BLOCKED', 'a'), record('', 'b'), record('too many source tokens', 'c'),
        record('first', 'one'), record(' FIRST ', 'different'), record('different', 'ONE'),
        record('second', 'two')], Tokenizer(), 2, 3, {'blocked'})
    assert rows == [{'en': 'first', 'hi': 'one'}, {'en': 'second', 'hi': 'two'}]
    assert counts == {'empty': 1, 'overlap_or_duplicate': 3, 'overlength': 1}


def test_sampling_fails_if_not_enough_eligible_rows():
    with pytest.raises(ValueError, match='Only 1 eligible'):
        select_rows([record('one', 'एक')], Tokenizer(), 2, 4)


def test_disjoint_checks_target_overlap_as_well_as_source():
    with pytest.raises(ValueError, match='overlapping hi'):
        assert_disjoint({'train': [{'en': 'a', 'hi': 'एक'}], 'test': [{'en': 'b', 'hi': 'एक'}]})
    assert_disjoint({'train': [{'en': 'a', 'hi': 'एक'}], 'test': [{'en': 'b', 'hi': 'दो'}]})


def test_fingerprint_and_normalization_are_stable():
    assert normalize('  A\t B  ') == 'a b'
    assert fingerprint([{'en': 'a', 'hi': 'b'}]) == fingerprint([{'hi': 'b', 'en': 'a'}])
    assert fingerprint([{'en': 'a', 'hi': 'b'}]) != fingerprint([{'en': 'c', 'hi': 'b'}])
