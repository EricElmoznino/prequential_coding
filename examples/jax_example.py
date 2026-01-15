import os

import jax
import numpy as np

from pcl_jax.data import ArrayDataset, PCLDataModule
from pcl_jax.model import MLP, SupervisedClassificationModel, SupervisedRegressionModel
from pcl_jax.training import PCLTask, Trainer, WandbLogger


def generate_debug_data(
    x_dim: int, discrete_y: bool, num_samples: int, seed: int
) -> ArrayDataset:
    rng = np.random.default_rng(seed)
    bound = 1.0 / np.sqrt(max(1, x_dim))
    w = rng.uniform(-bound, bound, size=(x_dim, 10)).astype(np.float32)
    b = rng.uniform(-bound, bound, size=(10,)).astype(np.float32)
    x = rng.standard_normal((num_samples, x_dim)).astype(np.float32)
    y = x @ w + b
    std = y.std(axis=0, keepdims=True)
    y = y + 0.2 * std * rng.standard_normal(size=y.shape).astype(np.float32)
    if discrete_y:
        y = y.argmax(axis=-1).astype(np.int32)
    else:
        y = y.astype(np.float32)
    return ArrayDataset(x, y)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--x_dim", type=int, default=10)
    parser.add_argument("--continuous_y", action="store_true")
    parser.add_argument("--num_samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    dataset = generate_debug_data(
        x_dim=args.x_dim,
        discrete_y=not args.continuous_y,
        num_samples=args.num_samples,
        seed=args.seed,
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

    base_model = MLP(
        layer_sizes=[args.x_dim, 128, 128, 10],
        activation=jax.nn.relu,
    )
    if args.continuous_y:
        model = SupervisedRegressionModel(model=base_model)
    else:
        model = SupervisedClassificationModel(model=base_model)

    task = PCLTask(
        model=model,
        interval_patience=20,
        interval_patience_tol=1e-3,
    )

    logger = WandbLogger(project="PCL-Debug", save_dir=args.save_dir)
    logger.log_hparams(vars(args))
    trainer = Trainer(
        logger=logger,
        max_epochs=-1,
        reload_dataloaders_every_n_epochs=1,
    )
    trainer.fit(
        model=task,
        datamodule=datamodule,
        rng=jax.random.PRNGKey(args.seed),
        data_rng=np.random.default_rng(args.seed),
    )
