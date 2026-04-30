"""Tests for core normalizer module."""

import numpy as np
import pytest
from src.core import Normalizer


def test_normalizer_fit():
    """Test Normalizer fit method."""
    data = np.random.randn(100, 162)
    n = Normalizer()
    n.fit(data)

    assert n.mean is not None
    assert n.std is not None
    assert n.is_fitted


def test_normalizer_transform():
    """Test Normalizer transform method."""
    data = np.random.randn(100, 162)
    n = Normalizer()
    n.fit(data)

    x = np.random.randn(10, 162)
    y = n.transform(x)

    assert y.shape == x.shape
    assert np.abs(y.mean()) < 0.1  # Approximately zero mean


def test_normalizer_inverse():
    """Test Normalizer inverse_transform method."""
    data = np.random.randn(100, 162)
    n = Normalizer()
    n.fit(data)

    x = np.random.randn(10, 162)
    y = n.transform(x)
    z = n.inverse_transform(y)

    assert np.allclose(x, z, atol=1e-5)


def test_normalizer_serialization():
    """Test Normalizer serialization/deserialization."""
    n = Normalizer()
    n.fit(np.random.randn(100, 162))

    d = n.to_dict()
    n2 = Normalizer.from_dict(d)

    assert np.allclose(n.mean, n2.mean)
    assert np.allclose(n.std, n2.std)


def test_normalizer_fit_transform():
    """Test Normalizer fit_transform convenience method."""
    data = np.random.randn(100, 162)
    n = Normalizer()

    y = n.fit_transform(data)

    assert y.shape == data.shape
    assert n.is_fitted
