from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LikelihoodModel(ABC, nn.Module):
    @abstractmethod
    def nll(self, x: Any, encode: bool) -> Tensor:
        """Return the negative log-likelihood of the model on some data.

        Args:
            x (Any): Batch of inputs.
            encode (bool): If True, return the code lengths. If False, return transformed code lengths used for loss (e.g., the average negative log-likelihood per element).

        Returns:
            Tensor: (batch_size,), negative log-likelihood for each input sample in the batch.
        """
        pass

    @abstractmethod
    def naive_code_length(self, x: Any) -> Tensor:
        """Return the naive code length (i.e., without the model) on some data.

        Args:
            x (Any): Batch of inputs.
        Returns:
            Tensor: (batch_size,), naive code length for each input sample in the batch.
        """
        pass


# ---------------------------------
# -------- Example models ---------
# ---------------------------------


class SupervisedRegressionModel(LikelihoodModel):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def nll(
        self,
        data: tuple[Tensor, Tensor],
        encode: bool = False,
    ) -> Tensor:
        x, y = data
        pred = self.model(x)
        error = F.mse_loss(pred, y, reduction="none")
        if encode:
            error = error.sum(dim=list(range(1, error.ndim)))
        else:
            error = error.mean(dim=list(range(1, error.ndim)))
        return error

    def naive_code_length(self, data: tuple[Tensor, Tensor]) -> Tensor:
        _, y = data
        pred = y.mean(dim=0, keepdim=True).expand_as(y)
        error = F.mse_loss(pred, y, reduction="none")
        error = error.sum(dim=list(range(1, error.ndim)))
        return error


class SupervisedClassificationModel(LikelihoodModel):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def nll(
        self,
        data: tuple[Tensor, Tensor],
        encode: bool = False,
    ) -> Tensor:
        x, y = data
        pred = self.model(x)
        error = F.cross_entropy(pred, y, reduction="none")
        if encode:
            error = error.sum(dim=list(range(1, error.ndim)))
        else:
            error = error.mean(dim=list(range(1, error.ndim)))
        return error

    def naive_code_length(self, data: tuple[Tensor, Tensor]) -> Tensor:
        _, y = data
        y_marginal = F.one_hot(y, num_classes=y.max() + 1).float().mean(dim=0)
        y_marginal = torch.distributions.Categorical(probs=y_marginal)
        error: Tensor = -y_marginal.log_prob(y)
        error = error.sum(dim=list(range(1, error.ndim)))
        return error
