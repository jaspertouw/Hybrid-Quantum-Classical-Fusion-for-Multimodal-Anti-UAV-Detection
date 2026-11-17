import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.thesis_experiments.neural_fusion import NeuralFusion, apply_gate, rotation, evidential_loss, evidential_output


def test_evidential_uniform_identities_and_losses():
    evidence = torch.zeros((2, 2), dtype=torch.float64)
    target = torch.tensor([0., 1.], dtype=torch.float64)
    p, u = evidential_output(evidence)
    torch.testing.assert_close(p, torch.full_like(p, 0.5))
    torch.testing.assert_close(u, torch.ones_like(u))
    assert float(evidential_loss(evidence, target, 1)) == pytest.approx(1.)
    assert float(evidential_loss(evidence, target, 1, "squared_error")) == pytest.approx(2 / 3)


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_circuit_parameter_count_and_expectation_bounds(depth):
    torch.manual_seed(11)
    model = NeuralFusion("vqc", depth=depth)
    assert sum(v.numel() for v in model.parameters()) == 12 * depth + 10
    q = model.quantum_features(torch.rand((4, 8), dtype=torch.float64))
    assert q.shape == (4, 4)
    assert bool((q.abs() <= 1 + 1e-12).all())


def test_isolated_qubit_matches_methodology_analytic_formula():
    model = NeuralFusion("vqc", depth=1, layout="none")
    x = torch.tensor([[0.3, 0.1, 0.4, 0.6, 0.8, 0.2, 0.3, 0.5]], dtype=torch.float64)
    with torch.no_grad():
        model.angles.zero_()
        model.angles[0, 0, 2] = 0.4
    r, t = math.pi / 4 + math.pi / 2 * 0.3, math.pi * 0.8
    expected = math.cos(r) * math.cos(0.4) - math.sin(r) * math.cos(t) * math.sin(0.4)
    assert float(model.quantum_features(x)[0, 0]) == pytest.approx(expected, abs=1e-12)


def test_circuit_input_and_parameter_gradients_match_finite_difference():
    torch.manual_seed(11)
    model = NeuralFusion("vqc", depth=2)
    x = torch.rand((1, 8), dtype=torch.float64, requires_grad=True)
    value = model.quantum_features(x)[0, 0]
    gx, ga = torch.autograd.grad(value, (x, model.angles))
    h = 1e-6
    with torch.no_grad():
        plus, minus = x.clone(), x.clone()
        plus[0, 4] += h
        minus[0, 4] -= h
        finite_input = (model.quantum_features(plus)[0, 0] - model.quantum_features(minus)[0, 0]) / (2 * h)
        old = model.angles[0, 0, 0].item()
        model.angles[0, 0, 0] = old + h
        upper = model.quantum_features(x)[0, 0].item()
        model.angles[0, 0, 0] = old - h
        lower = model.quantum_features(x)[0, 0].item()
        model.angles[0, 0, 0] = old
    assert float(gx[0, 4]) == pytest.approx(float(finite_input), abs=1e-7)
    assert float(ga[0, 0, 0]) == pytest.approx((upper - lower) / (2 * h), abs=1e-7)


@pytest.mark.parametrize("kind", ["mlp", "gate", "vqc"])
def test_models_finite_gradients_and_saved_predictions(tmp_path, kind):
    model = NeuralFusion(kind)
    x = torch.rand((6, 8), dtype=torch.float64)
    # Valid UAV/background/vacuity partitions for gate inputs.
    x[:, 0] = 0.4
    x[:, 1] = 0.3
    x[:, 4] = 0.5
    x[:, 5] = 0.2
    p, u, evidence = model(x)
    assert bool(((p >= 0) & (p <= 1) & (u >= 0) & (u <= 1)).all())
    p.sum().backward()
    assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
    if kind == "gate":
        assert evidence is None
    model.save(tmp_path / "model.pt")
    restored = NeuralFusion.load(tmp_path / "model.pt")
    np.testing.assert_array_equal(model.predict(x.numpy())[0], restored.predict(x.numpy())[0])


def test_ablation_flags_and_dropout_inputs_have_valid_expectations():
    x = torch.tensor([[0, 1, 0, 1, 0.5, 0.5, 0.5, 0.5]], dtype=torch.float64)
    for layout in ("none", "chain", "ring"):
        for reupload in (False, True):
            model = NeuralFusion("vqc", layout=layout, reupload=reupload)
            assert torch.isfinite(model.quantum_features(x)).all()


def test_cnot_chain_prepares_ghz_and_ordered_ring_has_expected_basis_support():
    model = NeuralFusion("vqc")
    state = torch.zeros((1, 16), dtype=torch.complex128)
    state[0, 0] = 1
    state = apply_gate(state, rotation("y", torch.tensor([math.pi / 2], dtype=torch.float64)), 0)
    for edge in ("cnot_0_1", "cnot_1_2", "cnot_2_3"):
        state = state[:, getattr(model, edge)]
    expected = torch.zeros_like(state)
    expected[0, 0] = expected[0, 15] = 1 / math.sqrt(2)
    torch.testing.assert_close(state, expected)
    state = state[:, model.cnot_3_0]
    expected.zero_()
    expected[0, 0] = expected[0, 7] = 1 / math.sqrt(2)
    torch.testing.assert_close(state, expected)
