import os
import time

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

import ml_collections

# from absl import logging
import wandb

from jaxpi.samplers import UniformSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint

import models
from utils import get_dataset

from math import sqrt


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    logger = Logger()
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # Problem setup
    Tmin = 0
    Tmax = 10.0
    Xmin = -10
    Xmax = 40

    n_t = 200
    n_x = 128

    x0 = 0.0
    sigma = 1.0
    k0 = 1.0

    A = 1.0
    hbar = 1.0

    m = 1.0

    Tcar = max(abs(Tmin), abs(Tmax))
    Xcar = max(abs(Xmin), abs(Xmax))
    Mcar = m

    Tmin /= Tcar
    Tmax /= Tcar
    Xmin /= Xcar
    Xmax /= Xcar

    x0 /= Xcar
    sigma /= Xcar
    k0 *= Xcar
    A *= sqrt(Xcar)
    hbar = hbar / (Mcar * Xcar**2) * Tcar
    m /= Mcar

    # Get  dataset
    psi_real, psi_imag, t_star, x_star, Tcar, Xcar = get_dataset(
        Tmin, Tmax, Xmin, Xmax, n_t, n_x, x0, sigma, k0, A, hbar, m, Tcar, Xcar
    )

    # Initial condition
    psi_real0 = psi_real[0, :]
    psi_imag0 = psi_imag[0, :]

    # Define domain
    t0 = t_star[0]
    t1 = t_star[-1]

    x0 = x_star[0]
    x1 = x_star[-1]

    dom = jnp.array([[t0, t1], [x0, x1]])

    # Initialize model
    model = models.QPINN(
        config, psi_real0, psi_imag0, t_star, x_star, hbar, m, Tcar, Xcar
    )

    res_sampler = iter(UniformSampler(dom, config.training.batch_size_per_device))

    evaluator = models.QPINNEvaluator(config, model)

    # jit warm up
    print("Waiting for JIT...")
    start_time = time.time()
    for step in range(config.training.max_steps):
        batch = next(res_sampler)
        model.state = model.step(model.state, batch)

        # Update weights
        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))

                log_dict = evaluator(state, batch, psi_real, psi_imag)
                wandb.log(log_dict, step)
                end_time = time.time()
                logger.log_iter(step, start_time, end_time, log_dict)
                start_time = end_time

        # Saving
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
                save_checkpoint(
                    model.state, ckpt_path, keep=config.saving.num_keep_ckpts
                )

    return model
