import math

import clem


def test_stack_embeddings_shape():
    padded = [[float(i) for i in range(256)] for _ in range(32)]
    t = clem.tensor((padded, 32))
    assert t.shape == (32, 256)


def test_embedding_lookup():
    table = clem.randn(10, 8, seed=1)
    idx = clem.tensor([[0.0, 2.0], [1.0, 3.0]])
    out = table[idx]
    assert out.shape == (2, 2, 8)


def test_positional_encoding_fill():
    max_seq_len = 8
    d_model = 4
    pos = clem.arange(max_seq_len, dtype="float32").reshape(max_seq_len, 1)
    div_term = clem.exp(-clem.arange(0, d_model, 2) / d_model * clem.log(clem.tensor(10000.0)))
    pe = clem.zeros(max_seq_len, d_model)
    pe[:, 0::2] = clem.sin(pos * div_term)
    pe[:, 1::2] = clem.cos(pos * div_term)
    assert pe.shape == (max_seq_len, d_model)
    assert pe[0, 0] == 0.0


def test_positionally_encode_embeddings():
    d_model = 8
    max_seq_len = 4
    batch = 2
    seq = 3
    embeddings = clem.randn(batch, seq, d_model, seed=2)
    e_prime = embeddings * math.sqrt(d_model)
    pos = clem.arange(max_seq_len, dtype="float32").reshape(max_seq_len, 1)
    div_term = clem.exp(-clem.arange(0, d_model, 2) / d_model * clem.log(clem.tensor(10000.0)))
    pe = clem.zeros(max_seq_len, d_model)
    pe[:, 0::2] = clem.sin(pos * div_term)
    pe[:, 1::2] = clem.cos(pos * div_term)
    sliced = pe[:seq, :]
    out = sliced + e_prime
    assert out.shape == (batch, seq, d_model)
