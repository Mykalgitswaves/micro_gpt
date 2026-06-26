import clem


def test_softmax_rows_sum_to_one(approx):
    t = clem.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    out = t.softmax(-1)
    for row in range(2):
        assert approx(sum(out[row, col].item() for col in range(3)), 1.0)


def test_layer_norm(approx):
    t = clem.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = clem.layer_norm_fn(t, [2])
    assert out.shape == (2, 2)
    flat = [out[i, j].item() for i in range(2) for j in range(2)]
    mean = sum(flat) / len(flat)
    assert approx(mean, 0.0,)


def test_gelu_at_zero(approx):
    t = clem.tensor([0.0])
    out = clem.gelu_fn(t)
    assert approx(out[0].item(), 0.0)


def test_dropout_seeded(approx):
    t = clem.tensor([1.0, 1.0, 1.0, 1.0])
    a = clem.dropout(t, p=0.5, seed=123)
    b = clem.dropout(t, p=0.5, seed=123)
    for i in range(4):
        assert approx(a[i].item(), b[i].item())


def test_cross_entropy_forward(approx):
    logits = clem.tensor([[0.0, 2.0], [2.0, 0.0]])
    targets = clem.tensor([1.0, 0.0])
    loss = clem.cross_entropy(logits, targets)
    assert loss.shape == ()
    assert loss.item() > 0.0


def test_cross_entropy_backward():
    logits = clem.tensor([[1.0, 2.0, 3.0]])
    targets = clem.tensor([2.0])
    logits.requires_grad_(True)
    loss = clem.cross_entropy(logits, targets)
    loss.backward()
    assert logits.grad is not None
    assert abs(sum(logits.grad[0, i].item() for i in range(3))) < 1e-4


def test_cross_entropy_ignore_index(approx):
    logits = clem.tensor([[0.0, 2.0], [100.0, -100.0]])
    targets = clem.tensor([1.0, 0.0])
    loss = clem.cross_entropy(logits, targets, ignore_index=0)
    expected = clem.cross_entropy(
        clem.tensor([[0.0, 2.0]]),
        clem.tensor([1.0]),
    )
    assert approx(loss.item(), expected.item())
