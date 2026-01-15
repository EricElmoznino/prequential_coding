from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp


class LikelihoodModel(ABC):
    @abstractmethod
    def init_params(self, rng: jax.Array) -> Any:
        """Initialize parameters for the model."""
        raise NotImplementedError

    @abstractmethod
    def apply(self, params: Any, x: jax.Array) -> jax.Array:
        """Forward pass for the model."""
        raise NotImplementedError

    @abstractmethod
    def nll(self, params: Any, data: Any, encode: bool) -> jax.Array:
        """Return negative log-likelihood per sample."""
        raise NotImplementedError

    @abstractmethod
    def naive_code_length(self, data: Any) -> jax.Array:
        """Return naive code length per sample."""
        raise NotImplementedError


def _init_layer_params(
    rng: jax.Array, in_dim: int, out_dim: int
) -> dict[str, jax.Array]:
    w_key, b_key = jax.random.split(rng)
    bound = 1.0 / jnp.sqrt(max(1, in_dim))
    weight = jax.random.uniform(
        w_key, (in_dim, out_dim), minval=-bound, maxval=bound, dtype=jnp.float32
    )
    bias = jax.random.uniform(
        b_key, (out_dim,), minval=-bound, maxval=bound, dtype=jnp.float32
    )
    return {"w": weight, "b": bias}


class MLP:
    """Simple MLP implemented with pure JAX."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
    ):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least input and output dims.")
        self.layer_sizes = tuple(int(size) for size in layer_sizes)
        self.activation = activation

    def init_params(self, rng: jax.Array) -> list[dict[str, jax.Array]]:
        keys = jax.random.split(rng, len(self.layer_sizes) - 1)
        params = []
        for key, (in_dim, out_dim) in zip(
            keys, zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        ):
            params.append(_init_layer_params(key, in_dim, out_dim))
        return params

    def apply(self, params: list[dict[str, jax.Array]], x: jax.Array) -> jax.Array:
        for layer in params[:-1]:
            x = self.activation(jnp.dot(x, layer["w"]) + layer["b"])
        final = params[-1]
        return jnp.dot(x, final["w"]) + final["b"]


class SupervisedRegressionModel(LikelihoodModel):
    def __init__(self, model: MLP):
        self.model = model

    def init_params(self, rng: jax.Array) -> Any:
        return self.model.init_params(rng)

    def apply(self, params: Any, x: jax.Array) -> jax.Array:
        return self.model.apply(params, x)

    def nll(
        self, params: Any, data: tuple[jax.Array, jax.Array], encode: bool
    ) -> jax.Array:
        x, y = data
        pred = self.apply(params, x)
        error = (pred - y) ** 2
        if error.ndim > 1:
            axes = tuple(range(1, error.ndim))
            error = error.sum(axis=axes) if encode else error.mean(axis=axes)
        return error

    def naive_code_length(self, data: tuple[jax.Array, jax.Array]) -> jax.Array:
        _, y = data
        pred = jnp.mean(y, axis=0, keepdims=True)
        pred = jnp.broadcast_to(pred, y.shape)
        error = (pred - y) ** 2
        if error.ndim > 1:
            error = error.sum(axis=tuple(range(1, error.ndim)))
        return error


class SupervisedClassificationModel(LikelihoodModel):
    def __init__(self, model: MLP):
        self.model = model

    def init_params(self, rng: jax.Array) -> Any:
        return self.model.init_params(rng)

    def apply(self, params: Any, x: jax.Array) -> jax.Array:
        return self.model.apply(params, x)

    def nll(
        self, params: Any, data: tuple[jax.Array, jax.Array], encode: bool
    ) -> jax.Array:
        x, y = data
        logits = self.apply(params, x)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        y = y.astype(jnp.int32).reshape((-1,))
        nll = -log_probs[jnp.arange(y.shape[0]), y]
        if nll.ndim > 1:
            axes = tuple(range(1, nll.ndim))
            nll = nll.sum(axis=axes) if encode else nll.mean(axis=axes)
        return nll

    def naive_code_length(self, data: tuple[jax.Array, jax.Array]) -> jax.Array:
        _, y = data
        y = y.astype(jnp.int32).reshape((-1,))
        num_classes = int(jnp.max(y)) + 1
        counts = jnp.bincount(y, length=num_classes)
        probs = counts / jnp.sum(counts)
        probs = jnp.clip(probs, 1e-12, 1.0)
        nll = -jnp.log(probs[y])
        return nll
