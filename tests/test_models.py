import numpy as np
from policydiff import models


def test_semantic_long_clause_preserves_all_chunks(monkeypatch):
    class Tokenizer:
        def num_special_tokens_to_add(self, pair=False):
            return 2
        def encode(self, text, **kwargs):
            if ',' in text:
                return text.split(',')
            return list(range(int(text)))
        def decode(self, ids, **kwargs):
            return ','.join(map(str, ids))
    class Model:
        tokenizer = Tokenizer()
        max_seq_length = 6
        chunks = []
        def encode(self, chunks, **kwargs):
            self.chunks = chunks
            return np.tile([1., 0.], (len(chunks), 1))
    model = Model()
    monkeypatch.setattr(models, 'embedding_model', lambda: model)
    vectors = models.semantic_vectors(['10', '2'])
    assert model.chunks == ['0,1,2,3', '4,5,6,7', '8,9', '0,1']
    assert vectors.shape == (2, 2)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1)
