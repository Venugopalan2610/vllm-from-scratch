"""Three functions for the challenge "measure it yourself".

Do not read this file before you finish the challenge. The point of the
challenge is to find the shape of each function from the outside, with
measurements only, the way you find the shape of a system that you did not
write.
"""
import torch

_FIXED = None


def _fixed():
    global _FIXED
    if _FIXED is None:
        _FIXED = torch.randn(128 * 2**20, device='cuda', dtype=torch.bfloat16)
    return _FIXED


def mystery_1(n):
    points = torch.randn(n, 64, device='cuda')
    return lambda: torch.cdist(points, points)


def mystery_2(n):
    small = torch.randn(n, device='cuda')
    return lambda: (_fixed().sum(), small.sum())


def mystery_3(n):
    values = torch.randn(n * 2**16, device='cuda')
    return lambda: values.sum()
