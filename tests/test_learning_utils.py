# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

import pytest

th = pytest.importorskip("torch")

from assume.reinforcement_learning.learning_utils import (  # noqa: E402
    _TORCH_BACKEND_CHECK,
    resolve_device,
)

pytestmark = pytest.mark.require_learning


def test_resolve_device_available(monkeypatch):
    """Test that a valid device which passes the check is returned as-is."""
    monkeypatch.setitem(_TORCH_BACKEND_CHECK, "mps", lambda d: True)

    device = resolve_device("mps")

    assert device == th.device("mps")


def test_resolve_device_unavailable(monkeypatch):
    """Test that a valid device which fails the backend check falls back to CPU."""
    monkeypatch.setitem(_TORCH_BACKEND_CHECK, "cuda", lambda d: False)

    device = resolve_device("cuda")

    assert device == th.device("cpu")


def test_resolve_device_unsupported_type(monkeypatch):
    """Test that a valid device type missing from _TORCH_BACKEND_CHECK falls back to CPU."""
    monkeypatch.delitem(_TORCH_BACKEND_CHECK, "meta", raising=False)

    device = resolve_device("meta")

    assert device == th.device("cpu")


def test_resolve_device_invalid_string():
    """Test that malformed device strings caught by torch.device fall back to CPU."""
    device = resolve_device("not_a_real_device_string_12345!@#")

    assert device == th.device("cpu")
