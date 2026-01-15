import os
import pickle
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pcl_jax.data import PCLDataModule, PCLDataset
from pcl_jax.model import LikelihoodModel


class BaseLogger:
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        pass

    def log_hparams(self, params: dict[str, Any]) -> None:
        pass

    def define_metric(self, name: str, step_metric: str | None = None) -> None:
        pass


class NullLogger(BaseLogger):
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        return None

    def log_hparams(self, params: dict[str, Any]) -> None:
        return None

    def define_metric(self, name: str, step_metric: str | None = None) -> None:
        return None


class WandbLogger(BaseLogger):
    def __init__(self, project: str, save_dir: str | None = None, **kwargs: Any):
        try:
            import wandb  # type: ignore
        except ImportError as exc:
            raise ImportError("wandb is required for WandbLogger.") from exc
        self._wandb = wandb
        self._run = wandb.init(project=project, dir=save_dir, **kwargs)

    @property
    def experiment(self):
        return self._run

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._wandb.log(metrics, step=step)

    def log_hparams(self, params: dict[str, Any]) -> None:
        self._run.config.update(params)

    def define_metric(self, name: str, step_metric: str | None = None) -> None:
        self._wandb.define_metric(name, step_metric=step_metric)


def _tree_zeros_like(tree: Any) -> Any:
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


class AdamState(NamedTuple):
    step: jax.Array
    m: Any
    v: Any


class Adam:
    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def init(self, params: Any) -> AdamState:
        return AdamState(
            step=jnp.array(0, dtype=jnp.int32),
            m=_tree_zeros_like(params),
            v=_tree_zeros_like(params),
        )

    def update(
        self, params: Any, grads: Any, state: AdamState
    ) -> tuple[Any, AdamState]:
        step = state.step + 1
        m = jax.tree_util.tree_map(
            lambda m_t, g_t: self.beta1 * m_t + (1.0 - self.beta1) * g_t,
            state.m,
            grads,
        )
        v = jax.tree_util.tree_map(
            lambda v_t, g_t: self.beta2 * v_t + (1.0 - self.beta2) * (g_t**2),
            state.v,
            grads,
        )
        bias_correction1 = 1.0 - jnp.power(self.beta1, step)
        bias_correction2 = 1.0 - jnp.power(self.beta2, step)
        m_hat = jax.tree_util.tree_map(lambda m_t: m_t / bias_correction1, m)
        v_hat = jax.tree_util.tree_map(lambda v_t: v_t / bias_correction2, v)
        params = jax.tree_util.tree_map(
            lambda p_t, m_t, v_t: p_t - self.lr * m_t / (jnp.sqrt(v_t) + self.eps),
            params,
            m_hat,
            v_hat,
        )
        return params, AdamState(step=step, m=m, v=v)


def _params_to_numpy(params: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: np.array(x), jax.device_get(params))


def _params_from_numpy(params: Any) -> Any:
    return jax.tree_util.tree_map(jnp.asarray, params)


class PCLTask:
    def __init__(
        self,
        model: LikelihoodModel,
        interval_patience: int = 15,
        interval_patience_tol: float = 0.0,
        lr: float = 1e-3,
        model_cache: str | None = None,
        include_initial_length: bool = True,
    ):
        """Builds the prequential coding curve by training models on incrementally more data."""
        self.model = model
        self.interval_patience = interval_patience
        self.interval_patience_tol = interval_patience_tol
        self.interval_epochs_since_improvement = 0
        self.interval_best_loss = float("inf")
        self.interval_errors: list[float] = []
        self.include_initial_length = include_initial_length

        self.model_cache = model_cache
        if model_cache is not None:
            if not os.path.exists(model_cache):
                raise ValueError(f"model_cache directory does not exist: {model_cache}")

        self.optimizer = Adam(lr=lr)
        self.params: Any | None = None
        self.opt_state: AdamState | None = None
        self.best_params: Any | None = None
        self._train_step = None
        self._eval_step = None
        self._encode_step = None
        self._logger: BaseLogger = NullLogger()
        self._datamodule: PCLDataModule | None = None
        self._rng: jax.Array | None = None
        self.should_stop = False

    def setup(
        self, rng: jax.Array, datamodule: PCLDataModule, logger: BaseLogger | None
    ) -> None:
        self._rng = rng
        self._datamodule = datamodule
        self._logger = logger or NullLogger()
        self._logger.define_metric("K/prequential curve", step_metric="data encoded")

        self.params = self.model.init_params(self._next_rng())
        self.opt_state = self.optimizer.init(self.params)

        def loss_fn(params: Any, batch: Any) -> jax.Array:
            return self.model.nll(params, batch, encode=False).mean()

        grad_fn = jax.value_and_grad(loss_fn)

        @jax.jit
        def train_step(params: Any, opt_state: AdamState, batch: Any):
            loss, grads = grad_fn(params, batch)
            params, opt_state = self.optimizer.update(params, grads, opt_state)
            return params, opt_state, loss

        @jax.jit
        def eval_step(params: Any, batch: Any):
            return self.model.nll(params, batch, encode=False).mean()

        @jax.jit
        def encode_step(params: Any, batch: Any):
            return self.model.nll(params, batch, encode=True).sum()

        self._train_step = train_step
        self._eval_step = eval_step
        self._encode_step = encode_step

    def _next_rng(self) -> jax.Array:
        if self._rng is None:
            raise RuntimeError("Task RNG is not initialized.")
        self._rng, key = jax.random.split(self._rng)
        return key

    def log(self, name: str, value: float, step: int | None = None) -> None:
        self._logger.log_metrics({name: float(value)}, step=step)

    def training_step(self, data: Any) -> float:
        assert self.params is not None
        assert self.opt_state is not None
        params, opt_state, loss = self._train_step(self.params, self.opt_state, data)
        self.params = params
        self.opt_state = opt_state
        return float(loss)

    def validation_step(self, data: Any) -> float:
        assert self.params is not None
        loss = self._eval_step(self.params, data)
        return float(loss)

    def on_train_start(self) -> None:
        dataset = self.dataset
        if dataset.prev_data_encoded != 0:
            raise RuntimeError("Dataset should start with prev_data_encoded == 0.")
        if self.include_initial_length:
            initial_length = float(jnp.sum(self.model.naive_code_length(dataset[:])))
            self.interval_errors.append(initial_length)
            for i in range(1, dataset.data_encoded + 1):
                self._logger.log_metrics(
                    {
                        "data encoded": float(i),
                        "K/prequential curve": initial_length / dataset.data_encoded,
                    }
                )

    def on_validation_epoch_start(self) -> None:
        self.dataset.set_mode("val")

    def on_validation_epoch_end(self) -> None:
        self.dataset.set_mode("train")

    def on_train_epoch_end(self, val_loss: float) -> None:
        if val_loss < self.interval_best_loss - self.interval_patience_tol:
            self.interval_best_loss = val_loss
            self.interval_epochs_since_improvement = 0
            if self.model_cache is None:
                self.best_params = self.params
            else:
                with open(os.path.join(self.model_cache, "best.pkl"), "wb") as f:
                    pickle.dump(_params_to_numpy(self.params), f)
        else:
            self.interval_epochs_since_improvement += 1

        if self.interval_epochs_since_improvement >= self.interval_patience:
            if self.model_cache is None:
                if self.best_params is None:
                    self.best_params = self.params
                self.params = self.best_params
            else:
                with open(os.path.join(self.model_cache, "best.pkl"), "rb") as f:
                    self.params = _params_from_numpy(pickle.load(f))

            if not self.dataset.done:
                self.dataset.increment_data_size()
                self.compute_length(mode="encode")
                self.interval_epochs_since_improvement = 0
                self.interval_best_loss = float("inf")
                self.reset_model_params()
            else:
                self.compute_length(mode="all_train")
                self.should_stop = True

    def compute_length(self, mode: str) -> None:
        dataset = self.dataset
        prev_mode = dataset.mode
        dataset.set_mode(mode)
        dataloader = self.datamodule.val_dataloader(shuffle=False)

        neg_logp = 0.0
        for data in dataloader:
            neg_logp += float(self._encode_step(self.params, data))

        if mode == "encode":
            naive = float(jnp.sum(self.model.naive_code_length(dataset[:])))
            neg_logp = min(neg_logp, naive)
            self.interval_errors.append(neg_logp)
            for i in range(dataset.prev_data_encoded + 1, dataset.data_encoded + 1):
                self._logger.log_metrics(
                    {
                        "data encoded": float(i),
                        "K/prequential curve": neg_logp / len(dataset),
                    }
                )
        else:
            k_data = float(sum(self.interval_errors))
            self.log("K/K(Data)", k_data)
            self.log("K/K(Data|f)", neg_logp)
            self.log("K/K(f)", k_data - neg_logp)

        dataset.set_mode(prev_mode)

    def reset_model_params(self) -> None:
        self.params = self.model.init_params(self._next_rng())
        self.opt_state = self.optimizer.init(self.params)

    @property
    def datamodule(self) -> PCLDataModule:
        if self._datamodule is None:
            raise RuntimeError("Datamodule is not initialized.")
        return self._datamodule

    @property
    def dataset(self) -> PCLDataset:
        dm = self.datamodule
        if dm.dataset is None:
            raise RuntimeError("Datamodule is not set up.")
        if not isinstance(dm.dataset, PCLDataset):
            raise RuntimeError("Datamodule dataset is not a PCLDataset.")
        return dm.dataset


class Trainer:
    def __init__(
        self,
        logger: BaseLogger | None = None,
        max_epochs: int = -1,
        reload_dataloaders_every_n_epochs: int = 1,
        log_every_n_steps: int | None = None,
    ):
        self.logger = logger or NullLogger()
        self.max_epochs = max_epochs
        self.reload_dataloaders_every_n_epochs = reload_dataloaders_every_n_epochs
        self.log_every_n_steps = log_every_n_steps

    def fit(
        self,
        model: PCLTask,
        datamodule: PCLDataModule,
        rng: jax.Array,
        data_rng: np.random.Generator | None = None,
    ) -> None:
        if self.reload_dataloaders_every_n_epochs != 1:
            raise ValueError("reload_dataloaders_every_n_epochs must be 1 for PCL.")

        datamodule.setup()
        model.setup(rng=rng, datamodule=datamodule, logger=self.logger)
        model.on_train_start()

        epoch = 0
        data_rng = data_rng or np.random.default_rng()

        while True:
            if self.max_epochs != -1 and epoch >= self.max_epochs:
                break

            train_loader = datamodule.train_dataloader(shuffle=True, rng=data_rng)
            train_losses: list[float] = []
            for step, batch in enumerate(train_loader, start=1):
                loss = model.training_step(batch)
                train_losses.append(loss)
                if self.log_every_n_steps and step % self.log_every_n_steps == 0:
                    model.log("training/train_loss", loss)

            if train_losses:
                model.log("training/train_loss", float(np.mean(train_losses)))

            model.on_validation_epoch_start()
            val_loader = datamodule.val_dataloader(shuffle=False)
            val_losses = [model.validation_step(batch) for batch in val_loader]
            mean_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
            model.log("training/val_loss", mean_val_loss)
            model.on_validation_epoch_end()

            model.on_train_epoch_end(mean_val_loss)
            if model.should_stop:
                break

            epoch += 1
