import json
from pathlib import Path
import shutil

import pytest

from scripts.verify_training_evidence import verify


EVIDENCE = Path(__file__).resolve().parents[1] / 'submission/results/iitb_3000_seed42'


def test_colab_evidence_recomputes_scores_and_checkpoint():
    result = verify(EVIDENCE)
    assert result['status'] == 'verified'
    assert result['selected_epoch'] == 1
    assert result['recomputed_scores']['fine_tuned']['bleu'] < result['recomputed_scores']['pretrained']['bleu']


def test_verifier_rejects_invented_score(tmp_path):
    folder = tmp_path / 'evidence'
    shutil.copytree(EVIDENCE, folder)
    path = folder / 'comparison.json'
    data = json.loads(path.read_text())
    data['fine_tuned']['bleu'] += 5
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(AssertionError):
        verify(folder)
