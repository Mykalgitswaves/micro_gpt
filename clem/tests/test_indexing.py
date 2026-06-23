import clem


def test_embedding_lookup():
    table = clem.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    idx = clem.tensor([0.0, 2.0])
    out = table[idx]
    assert out.shape == (2, 2)


def test_slice_even_odd_columns():
    t = clem.zeros(3, 4)
    even = clem.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    odd = clem.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    t[:, 0::2] = even
    t[:, 1::2] = odd
    assert t[0, 0].item() == 1.0
    assert t[0, 1].item() == 7.0
    assert t[0, 2] == 2.0


def test_row_slice():
    pe = clem.zeros(256, 64)
    sliced = pe[:32, :]
    assert sliced.shape == (32, 64)
