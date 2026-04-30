"""Core normalization utilities for TSL-51."""

import numpy as np
from typing import Optional


class Normalizer:
    """Z-score normalizer with statistics tracking."""

    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        self.mean = mean
        self.std = std

    @property
    def is_fitted(self) -> bool:
        return self.mean is not None and self.std is not None

    def fit(self, features: np.ndarray) -> "Normalizer":
        """Fit normalizer to training data."""
        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0)
        self.std = np.where(self.std == 0, 1.0, self.std)  # Avoid div by zero
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted statistics."""
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")
        return (features - self.mean) / self.std

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(features)
        return self.transform(features)

    def inverse_transform(self, normalized: np.ndarray) -> np.ndarray:
        """Reverse normalization."""
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before inverse_transform")
        return normalized * self.std + self.mean

    def to_dict(self) -> dict:
        """Serialize to dict for checkpoint."""
        return {
            'mean': self.mean.tolist() if self.mean is not None else None,
            'std': self.std.tolist() if self.std is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Normalizer":
        """Deserialize from dict."""
        mean = np.array(data['mean']) if data.get('mean') is not None else None
        std = np.array(data['std']) if data.get('std') is not None else None
        return cls(mean=mean, std=std)