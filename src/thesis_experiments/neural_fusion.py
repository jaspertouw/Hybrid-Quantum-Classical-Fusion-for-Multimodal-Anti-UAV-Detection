"""Evidential MLP, inherited belief gate and exact four-qubit statevector VQC."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .metrics import sequence_scores


def evidential_output(evidence):
    alpha = evidence + 1
    strength = alpha.sum(dim=-1)
    return alpha[:, 0] / strength, 2 / strength


def evidential_loss(evidence, target, epoch, kind="cross_entropy"):
    alpha = evidence + 1
    strength = alpha.sum(dim=1, keepdim=True)
    y = torch.stack((target, 1 - target), dim=1)
    if kind == "cross_entropy":
        fit = (y * (torch.digamma(strength) - torch.digamma(alpha))).sum(dim=1)
    elif kind == "squared_error":
        p = alpha / strength
        fit = ((y - p) ** 2 + p * (1 - p) / (strength + 1)).sum(dim=1)
    else:
        raise ValueError("unknown evidential loss")
    adjusted = y + (1 - y) * alpha
    kl = torch.distributions.kl_divergence(torch.distributions.Dirichlet(adjusted),
                                          torch.distributions.Dirichlet(torch.ones_like(adjusted)))
    return (fit + min(1., epoch / 10) * kl).mean()


def rotation(axis, angle):
    c, s = torch.cos(angle / 2), torch.sin(angle / 2)
    z = torch.zeros_like(c)
    if axis == "y":
        values = (c, -s, s, c)
    elif axis == "x":
        values = (c, -1j * s, -1j * s, c)
    elif axis == "z":
        values = (torch.exp(-0.5j * angle), z, z, torch.exp(0.5j * angle))
    else:
        raise ValueError("unknown rotation axis")
    return torch.stack(values, dim=-1).reshape(-1, 2, 2).to(torch.complex128)


def apply_gate(state, matrix, qubit):
    shaped = state.reshape(len(state), 2 ** qubit, 2, 2 ** (3 - qubit))
    return torch.einsum("boi,blir->blor", matrix.expand(len(state), -1, -1), shaped).reshape(-1, 16)


class NeuralFusion(nn.Module):
    def __init__(self, kind, *, widths=(16,), depth=2, layout="ring", reupload=True):
        super().__init__()
        self.config = {"kind": kind, "widths": list(widths), "depth": depth,
                       "layout": layout, "reupload": reupload}
        self.kind = kind
        if kind == "mlp":
            layers, previous = [], 8
            for width in widths:
                layers.extend((nn.Linear(previous, width), nn.ReLU()))
                previous = width
            layers.append(nn.Linear(previous, 2))
            self.head = nn.Sequential(*layers)
        elif kind == "gate":
            self.head = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
        elif kind == "vqc":
            if depth not in (1, 2, 3, 4) or layout not in ("ring", "chain", "none"):
                raise ValueError("unsupported circuit architecture")
            self.angles = nn.Parameter(torch.empty(depth, 4, 3, dtype=torch.float64))
            nn.init.normal_(self.angles, std=0.1)
            self.head = nn.Linear(4, 2)
            bits = torch.arange(16)
            self.register_buffer("z_signs", torch.stack([1 - 2 * ((bits >> (3 - q)) & 1) for q in range(4)]).double())
            for control, target in ((0, 1), (1, 2), (2, 3), (3, 0)):
                permutation = bits ^ ((((bits >> (3 - control)) & 1)) << (3 - target))
                self.register_buffer(f"cnot_{control}_{target}", permutation)
        else:
            raise ValueError("unknown neural fusion kind")
        self.double()
        for layer in self.head.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def quantum_features(self, x):
        state = torch.zeros((len(x), 16), dtype=torch.complex128, device=x.device)
        state[:, 0] = 1
        for block in range(self.config["depth"]):
            if block == 0 or self.config["reupload"]:
                for q in range(4):
                    state = apply_gate(state, rotation("y", math.pi / 4 + math.pi / 2 * x[:, q]), q)
                    state = apply_gate(state, rotation("z", math.pi * x[:, q + 4]), q)
            for q in range(4):
                for axis, angle in zip(("x", "z", "y"), self.angles[block, q]):
                    state = apply_gate(state, rotation(axis, angle.reshape(1)), q)
            if self.config["layout"] != "none":
                edges = [(0, 1), (1, 2), (2, 3)] + ([(3, 0)] if self.config["layout"] == "ring" else [])
                for control, target in edges:
                    state = state[:, getattr(self, f"cnot_{control}_{target}")]
        probabilities = state.abs() ** 2
        return probabilities @ self.z_signs.T

    def forward(self, x):
        if self.kind == "gate":
            br = torch.stack((x[:, 0], (1 - x[:, 0] - x[:, 1]).clamp(min=0)), dim=1)
            bt = torch.stack((x[:, 4], (1 - x[:, 4] - x[:, 5]).clamp(min=0)), dim=1)
            g = self.head(torch.cat((br, bt), dim=1))
            belief = g * br + (1 - g) * bt
            u = (1 - belief.sum(dim=1)).clamp(0, 1)
            return belief[:, 0] + u / 2, u, None
        representation = self.quantum_features(x) if self.kind == "vqc" else x
        evidence = F.softplus(self.head(representation))
        p, u = evidential_output(evidence)
        return p, u, evidence

    def predict(self, x, batch_size=1024):
        self.eval()
        probabilities, uncertainty = [], []
        with torch.no_grad():
            for offset in range(0, len(x), batch_size):
                p, u, _ = self(torch.as_tensor(x[offset:offset + batch_size], dtype=torch.float64))
                probabilities.extend(p.numpy())
                uncertainty.extend(u.numpy())
        return np.array(probabilities), np.array(uncertainty)

    def save(self, path):
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(**saved["config"])
        model.load_state_dict(saved["state_dict"])
        return model


def train_neural(model, x, labels, validation_x, validation_data, *, epochs=100, patience=10,
                 learning_rate=0.001, seed=11, loss_kind="cross_entropy"):
    if epochs < 1 or patience < 1 or learning_rate <= 0 or not len(x) or len(labels) != len(x):
        raise ValueError("invalid training limits or support")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    features = torch.as_tensor(x, dtype=torch.float64)
    target = torch.as_tensor(labels.astype(float), dtype=torch.float64)
    history, best, checkpoint, stale, best_epoch = [], None, None, 0, None
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.
        for batch in torch.randperm(len(features), generator=generator).split(128):
            optimizer.zero_grad()
            p, _, evidence = model(features[batch])
            loss = (F.binary_cross_entropy(p, target[batch]) if model.kind == "gate" else
                    evidential_loss(evidence, target[batch], epoch, loss_kind))
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite neural training loss")
            loss.backward()
            if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):
                raise FloatingPointError("nonfinite neural gradients")
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        prediction, _ = model.predict(validation_x)
        acc = float(sequence_scores(validation_data, prediction)[0].mean())
        brier = float(np.mean((prediction - validation_data["label"]) ** 2))
        criterion = (acc, -brier)
        history.append({"epoch": epoch, "loss": total / len(x), "validation_edtc_at_05": acc,
                        "validation_brier": brier})
        if best is None or criterion > best:
            best, checkpoint, stale = criterion, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
            best_epoch = epoch
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(checkpoint)
    return {"history": history, "seconds": time.perf_counter() - started,
            "parameters": sum(p.numel() for p in model.parameters()), "epochs": len(history),
            "best_epoch": best_epoch, "stop_reason": "patience" if stale >= patience else "epoch_limit",
            "seed": seed, "learning_rate": learning_rate, "loss": "bce" if model.kind == "gate" else loss_kind}
