"""Backend interface and normalized outputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ModelBackend(ABC):
    """Common interface implemented by local and remote model backends."""

    @property
    @abstractmethod
    def capabilities(self) -> dict[str, bool]:
        """Return support for log-likelihood scoring and generation."""

    @property
    @abstractmethod
    def scoring_method(self) -> str:
        """Stable identifier describing the backend's log-likelihood method."""

    @abstractmethod
    def score_options(self, prompts: list[str], n_options: int) -> list[list[float]]:
        """Return one score per option for each prompt."""

    @abstractmethod
    def generate(self, prompts: list[str], max_new_tokens: int = 8) -> list[str]:
        """Generate one response for each prompt."""

    @property
    def metadata(self) -> dict[str, Any]:
        """Return serializable model and runtime metadata."""
        return {"scoring_method": self.scoring_method}
