import pytest

torch = pytest.importorskip("torch")


def test_circular_padding_matches_dit360_reference():
    from editpano.generation import circular_pad_tokens, circular_unpad_tokens

    x = torch.randn(2, 8 * 16, 4)
    padded = circular_pad_tokens(x, 8, 16)
    grid = x.reshape(2, 8, 16, 4)
    reference = torch.cat([grid[:, :, -1:], grid, grid[:, :, 0:1]], dim=2).reshape(2, -1, 4)
    assert torch.equal(padded, reference)
    assert torch.equal(circular_unpad_tokens(padded, 8, 16), x)
    ids = torch.randn(8 * 16, 3)
    assert circular_pad_tokens(ids, 8, 16).shape == (8 * 18, 3)
