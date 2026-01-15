import os
import warnings

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor, nn
from torch.utils.data import Dataset, TensorDataset

from pcl_pytorch.data import PCLDataModule
from pcl_pytorch.model import SupervisedClassificationModel, SupervisedRegressionModel
from pcl_pytorch.training import PCLTask

warnings.filterwarnings("ignore", message=".*does not have many workers.*")


@torch.inference_mode()
def generate_debug_data(x_dim: int, discrete_y: bool, num_samples: int) -> Dataset:
    f = nn.Linear(x_dim, 10).eval()
    x = torch.randn(num_samples, x_dim)
    y: Tensor = f(x)
    std = y.std(dim=0, keepdim=True)
    y += 0.2 * std * torch.randn_like(y)
    if discrete_y:
        y = y.argmax(dim=-1)
    return TensorDataset(x, y)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--x_dim", type=int, default=10)
    parser.add_argument("--continuous_y", action="store_true")
    parser.add_argument("--num_samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    dataset = generate_debug_data(
        x_dim=args.x_dim,
        discrete_y=not args.continuous_y,
        num_samples=args.num_samples,
    )

    datamodule = PCLDataModule(
        dataset=dataset,
        min_data_size=10,
        val_ratio=0.1,
        # data_increment=10,
        data_increment_n_log_intervals=30,
        batch_size=100,
        num_workers=0,
    )
    model = nn.Sequential(
        nn.Linear(args.x_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    if args.continuous_y:
        model = SupervisedRegressionModel(model=model)
    else:
        model = SupervisedClassificationModel(model=model)
    task = PCLTask(
        model=model,
        interval_patience=20,
        interval_patience_tol=1e-3,
    )

    logger = WandbLogger(project="PCL-Debug", save_dir=args.save_dir)
    logger.experiment.config.update(vars(args))
    trainer = Trainer(
        logger=logger,
        callbacks=None,
        enable_checkpointing=False,
        max_epochs=-1,
        num_sanity_val_steps=0,
        reload_dataloaders_every_n_epochs=1,  # NOTE: IMPORTANT FOR PCL DATAMODULE TO WORK!
    )
    trainer.fit(model=task, datamodule=datamodule)
