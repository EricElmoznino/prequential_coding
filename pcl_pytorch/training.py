import os
from copy import deepcopy

import torch
from lightning import LightningModule

from pcl_pytorch.data import PCLDataModule, PCLDataset
from pcl_pytorch.model import LikelihoodModel


class PCLTask(LightningModule):
    def __init__(
        self,
        model: LikelihoodModel,
        interval_patience: int = 15,
        interval_patience_tol: float = 0.0,
        lr: float = 1e-3,
        model_cache: str | None = None,
        include_initial_length: bool = True,
    ):
        """Builds the prequential coding curve by training models on incrementally more data.

        Args:
            model (LikelihoodModel): A model that can return a negative log-likelihood.
            interval_patience (int, optional): For some intermediate dataset, number of epochs with no validation improvement past which we stop training and increase the dataset size. Defaults to 15.
            interval_patience_tol (float, optional): Quantity below which we consider no improvement in validation performance. Defaults to 0.0.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            model_cache (str | None, optional): A directory in which to save the best model so far across epochs on disk, if model size is large. Defaults to None, in which case best model is kept in RAM.
            include_initial_length (bool, optional): Whether to include the naive code length of the initial dataset before training. Defaults to True.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model

        self.interval_patience = interval_patience
        self.interval_patience_tol = interval_patience_tol
        self.interval_epochs_since_improvement = 0
        self.interval_best_loss = torch.inf
        self.interval_errors: list[float] = []

        self.model_cache = model_cache
        if model_cache is None:
            self.best_model = None
        else:
            assert os.path.exists(model_cache)

    def training_step(self, data, batch_idx):
        loss = self.model.nll(data, encode=False).mean()
        self.log(
            "training/train_loss", loss, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, data, batch_idx):
        loss = self.model.nll(data, encode=False).mean()
        self.log("training/val_loss", loss, prog_bar=True)
        return loss

    def on_train_start(self):
        # NOTE: IMPORTANT FOR PCL DATAMODULE TO WORK!
        assert self.trainer is not None
        assert hasattr(self.trainer, "reload_dataloaders_every_n_epochs")
        assert self.trainer.reload_dataloaders_every_n_epochs == 1

        # Compute initial length if required
        dataset = self.dataset
        assert dataset.prev_data_encoded == 0
        if self.hparams.include_initial_length:
            initial_length = self.model.naive_code_length(dataset[:]).sum().cpu().item()
            self.interval_errors.append(initial_length)
            for i in range(1, dataset.data_encoded + 1):
                self.logger.log_metrics(
                    {
                        "data encoded": float(i),
                        "K/prequential curve": initial_length / dataset.data_encoded,
                    },
                    step=i,
                )

    def on_validation_epoch_start(self):
        self.dataset.set_mode("val")

    def on_validation_epoch_end(self):
        self.dataset.set_mode("train")

    def on_train_epoch_end(self):
        # Check for improvement on validation set
        loss = self.trainer.callback_metrics["training/val_loss"]
        if loss < self.interval_best_loss - self.interval_patience_tol:
            self.interval_best_loss = loss
            self.interval_epochs_since_improvement = 0
            if self.model_cache is None:
                self.best_model = deepcopy(self.model.state_dict())
            else:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.model_cache, "best.pt"),
                )
        else:
            self.interval_epochs_since_improvement += 1

        # If no improvement, stop training on this current set of data
        if self.interval_epochs_since_improvement >= self.interval_patience:
            if self.model_cache is None:
                self.model.load_state_dict(self.best_model)
            else:
                self.model.load_state_dict(
                    torch.load(os.path.join(self.model_cache, "best.pt"))
                )

            # Encode the next increment of data
            if not self.dataset.done:
                self.dataset.increment_data_size()
                self.compute_length(mode="encode")
                self.interval_epochs_since_improvement = 0
                self.interval_best_loss = torch.inf
                self.reset_model_params()
                self.trainer.optimizers = [self.configure_optimizers()]
            # Get final loss across the whole dataset
            else:
                self.compute_length(mode="all_train")
                self.trainer.should_stop = True

    @torch.inference_mode()
    def compute_length(self, mode: str):
        dataset = self.dataset
        prev_mode = dataset.mode
        dataset.set_mode(mode)
        dataloader = self.datamodule.val_dataloader()

        neg_logp = 0.0
        for data in dataloader:
            data = move_to_device(data, self.device)
            neg_logp += self.model.nll(data, encode=True).sum().cpu().item()

        if mode == "encode":
            neg_logp = min(
                neg_logp,
                self.model.naive_code_length(dataset[:]).sum().cpu().item(),
            )
            self.interval_errors.append(neg_logp)
            for i in range(dataset.prev_data_encoded + 1, dataset.data_encoded + 1):
                self.logger.log_metrics(
                    {
                        "data encoded": float(i),
                        "K/prequential curve": neg_logp / len(dataset),
                    },
                    step=i,
                )
        else:
            k_data = sum(self.interval_errors)
            self.log("K/K(Data)", k_data)
            self.log("K/K(Data|f)", neg_logp)
            self.log("K/K(f)", k_data - neg_logp)

        self.dataset.set_mode(prev_mode)

    def reset_model_params(self):
        def reset_module(module):
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        self.model.apply(reset_module)

    @property
    def datamodule(self) -> PCLDataModule:
        dm = self.trainer.datamodule
        assert isinstance(dm, PCLDataModule)
        return dm

    @property
    def dataset(self) -> PCLDataset:
        dm = self.datamodule
        assert isinstance(dm.dataset, PCLDataset)
        return dm.dataset

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)


def move_to_device(obj, device):
    """Recursively move any tensors in `obj` to `device`."""
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        t = type(obj)
        return t(move_to_device(v, device) for v in obj)
    else:
        return obj
