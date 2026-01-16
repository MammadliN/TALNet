# Minimal tests for pooling operators.

import numpy as np
import torch

from pooling import AutoPool1D, PowerPool1D, BetaExpPool1D


def test_autopool_matches_numpy(init: str) -> None:
    torch.manual_seed(0)
    x = torch.from_numpy((np.random.randn(1, 3, 10) ** 2).astype(np.float32))

    ap = AutoPool1D(axis=-1, init=init)
    y_torch = ap(x)

    assert tuple(y_torch.shape) == (1, 3)

    x_np = x.numpy()
    if init == "zeros":
        scaled = np.zeros_like(x_np)
    elif init == "ones":
        scaled = x_np
    else:
        raise ValueError

    max_val = np.max(scaled, axis=-1, keepdims=True)
    softmax = np.exp(scaled - max_val)
    weights = softmax / np.sum(softmax, axis=-1, keepdims=True)
    y_np = np.sum(x_np * weights, axis=-1)

    assert np.allclose(y_np, y_torch.detach().numpy(), atol=1e-6)


def test_autopool() -> None:
    test_autopool_matches_numpy("zeros")
    test_autopool_matches_numpy("ones")


def test_powerpool_basic() -> None:
    torch.manual_seed(0)
    x = torch.rand(2, 5, 7)
    pp = PowerPool1D(axis=1, init_n=1.0)
    y = pp(x)
    assert tuple(y.shape) == (2, 7)
    assert torch.isfinite(y).all()


def test_betaexp_pool_basic() -> None:
    torch.manual_seed(0)
    x = torch.rand(2, 5, 7)
    bep = BetaExpPool1D(axis=1, init_alpha=0.0, init_beta=1.0)
    y = bep(x)
    assert tuple(y.shape) == (2, 7)
    assert torch.isfinite(y).all()


if __name__ == "__main__":
    test_autopool()
    test_powerpool_basic()
    test_betaexp_pool_basic()
    print("All tests passed.")
