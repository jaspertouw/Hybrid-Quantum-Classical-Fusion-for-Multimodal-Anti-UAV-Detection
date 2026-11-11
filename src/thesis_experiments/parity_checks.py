"""Small numerical comparison used by the GPU parity diagnostic."""

import math


def compare_values(reference, actual, *, rtol=1e-4, atol=1e-5):
    if len(reference) != len(actual) or not reference:
        raise ValueError("parity values must have equal non-zero length")
    if not all(math.isfinite(value) for value in (*reference, *actual)):
        raise ValueError("non-finite parity values")
    differences = [abs(a - b) for a, b in zip(reference, actual)]
    # Same asymmetric tolerance convention as torch.allclose(reference, actual).
    limits = [atol + rtol * abs(value) for value in actual]
    return {
        "passed": all(error <= limit for error, limit in zip(differences, limits)),
        "max_absolute_error": max(differences),
        "max_tolerance_ratio": max(error / limit for error, limit in zip(differences, limits)),
    }
