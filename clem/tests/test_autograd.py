import clem


def _finite_diff_grad(fn, x, eps=1e-4):
    base = fn(x)
    grads = []
    flat = list(x)
    for i in range(len(flat)):
        xp = flat.copy()
        xm = flat.copy()
        xp[i] += eps
        xm[i] -= eps
        fp = fn(clem.tensor(xp))
        fm = fn(clem.tensor(xm))
        grads.append((fp[0] - fm[0]) / (2 * eps))
    return grads


def test_mul_backward():
    a = clem.tensor([2.0])
    b = clem.tensor([3.0])
    a.requires_grad_(True)
    b.requires_grad_(True)
    c = a * b
    c.backward()
    assert a.grad is not None
    assert abs(a.grad[0].item() - 3.0) < 1e-3
    assert abs(b.grad[0].item() - 2.0) < 1e-3


def test_exp_backward():
    x = clem.tensor([1.0])
    x.requires_grad_(True)
    y = clem.exp(x)
    y.backward()
    assert abs(x.grad[0].item() - 2.7182817) < 1e-3


def test_matmul_backward():
    a = clem.tensor([[1.0, 2.0]])
    b = clem.tensor([[3.0], [4.0]])
    a.requires_grad_(True)
    b.requires_grad_(True)
    c = a.matmul(b)
    c.backward()
    assert a.grad is not None
    assert b.grad is not None


def test_cross_entropy_reduces_loss():
    logits = clem.tensor([[2.0, 0.0], [2.0, 0.0]])
    targets = clem.tensor([0.0, 0.0])
    logits.requires_grad_(True)
    lr = 0.5

    for _ in range(5):
        loss = clem.cross_entropy(logits, targets)
        logits.zero_grad()
        loss.backward()
        rows = []
        for row in range(2):
            rows.append(
                [
                    logits[row, i].item() - lr * logits.grad[row, i].item()
                    for i in range(2)
                ]
            )
        logits = clem.tensor(rows)
        logits.requires_grad_(True)

    final_loss = clem.cross_entropy(logits, targets)
    assert final_loss.item() < 0.5


def test_sgd_reduces_loss():
    x = clem.tensor([[1.0], [2.0]])
    y = clem.tensor([[3.0], [5.0]])
    w = clem.tensor([[0.0]])
    w.requires_grad_(True)
    lr = 0.1

    for _ in range(3):
        pred = x.matmul(w)
        loss = ((pred - y) * (pred - y)).sum()
        w.zero_grad()
        loss.backward()
        w_data = w.item()
        g = w.grad.item()
        w = clem.tensor([[w_data - lr * g]])
        w.requires_grad_(True)

    final_pred = x.matmul(w)
    err = sum((final_pred[i, 0].item() - y[i, 0].item()) ** 2 for i in range(2))
    assert err < 5.0


def test_where_backward():
    cond = clem.tensor([1.0, 0.0])
    x = clem.tensor([3.0, 3.0])
    y = clem.tensor([1.0, 2.0])
    x.requires_grad_(True)
    y.requires_grad_(True)
    out = clem.where(cond, x, y)
    out.sum().backward()
    assert abs(x.grad[0].item() - 1.0) < 1e-3
    assert abs(x.grad[1].item() - 0.0) < 1e-3
    assert abs(y.grad[0].item() - 0.0) < 1e-3
    assert abs(y.grad[1].item() - 1.0) < 1e-3


def test_masked_fill_backward():
    t = clem.tensor([1.0, 2.0, 3.0])
    mask = clem.tensor([1.0, 0.0, 1.0])
    t.requires_grad_(True)
    out = t.masked_fill(mask, 5.0)
    out.sum().backward()
    assert abs(t.grad[0].item() - 0.0) < 1e-3
    assert abs(t.grad[1].item() - 1.0) < 1e-3
    assert abs(t.grad[2].item() - 0.0) < 1e-3
