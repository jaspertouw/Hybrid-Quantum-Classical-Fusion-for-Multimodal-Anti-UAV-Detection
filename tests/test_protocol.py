import json
from pathlib import Path

from scripts.validate_protocol import canonical_membership, load_protocol, validate_dataset


PROTOCOL_PATH = Path("experiments/protocol/v1.json")


def test_frozen_protocol_is_valid():
    protocol = load_protocol(PROTOCOL_PATH)
    assert protocol["protocol"]["version"] == "1.0.0"
    assert protocol["dataset"]["expected_sequence_counts"] == {
        "train": 160,
        "val": 67,
        "test": 91,
    }


def test_protocol_is_machine_readable_json():
    with PROTOCOL_PATH.open(encoding="utf-8") as handle:
        assert isinstance(json.load(handle), dict)


def test_membership_hash_is_order_independent(tmp_path):
    (tmp_path / "z-sequence").mkdir()
    (tmp_path / "a-sequence").mkdir()
    count, digest = canonical_membership(tmp_path)
    assert count == 2
    assert digest == "885da20c723ea987d40830d52924c5fe669de657451b4af9bc5e15a7eddadbd1"
