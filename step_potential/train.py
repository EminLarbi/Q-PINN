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
from jaxpi.utils import save_checkpoint, restore_checkpoint

import models
from utils import get_dataset

from math import sqrt


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    logger = Logger()

    # Problem setup
    Tmin = 0
    Tmax = 5.0
    Xmin = -15
    Xmax = 15

    n_t = Tmax * 10
    n_x = Xmax * 25 * 2

    x0 = 2.0
    sigma = 0.2
    k0 = -5

    A = 1.0
    hbar = 1.0
    k = 1
    m = 1.0

    e = 0.0223607
    e = 1

    Tcar = max(abs(Tmin), abs(Tmax))
    Xcar = max(abs(Xmin), abs(Xmax))
    Mcar = m
    Qcar = e
    x0 /= Xcar
    sigma /= Xcar
    k0 *= Xcar
    A *= sqrt(Xcar)

    hbar = hbar / (Mcar * Xcar**2) * Tcar
    m /= Mcar
    e /= Qcar

    k = k * Qcar**2 / (Mcar * Xcar**3) * Tcar**2

    Tmin /= Tcar
    Tmax /= Tcar
    Xmin /= Xcar
    Xmax /= Xcar
    # Get  dataset
    psi_real, psi_imag, t_star, x_star, Tcar, Xcar, psi_real0, psi_imag0 = get_dataset(
        Tmin,
        Tmax,
        Xmin,
        Xmax,
        n_t,
        n_x,
        x0,
        sigma,
        k0,
        A,
        hbar,
        m,
        e,
        k,
        Tcar,
        Xcar,
    )

    # Define domain
    t0 = t_star[0]
    t1 = t_star[-1]

    x0 = x_star[0]
    x1 = x_star[-1]

    dom = jnp.array([[t0, t1], [x0, x1]])

    # Initialize model
    model = models.QPINN(
        config,
        psi_real0,
        psi_imag0,
        t_star,
        x_star,
        hbar,
        m,
        e,
        k,
        Tcar,
        Xcar,
        Mcar,
    )

    res_sampler = iter(UniformSampler(dom, config.training.batch_size_per_device))

    batch = next(res_sampler)
    model.state = model.step(model.state, batch)

    #ckpt_path = "/content/drive/MyDrive/JAX-PI_1D/default_12/ckpt"

    # Guardamos el estado previo para rollback en caso de error
    state_backup = model.state
    try:
        model.state = restore_checkpoint(model.state, ckpt_path)
        # Replicar el estado en cada dispositivo para cumplir con el pmap
        model.state = jax.device_put_replicated(model.state, jax.local_devices())
    except Exception as e:
        model.state = state_backup
        # Rollback: se restaura el estado previo
        # Se lanza un error estructurado con información en inglés
        
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

                end_time = time.time()
                logger.log_iter(step, start_time, end_time, log_dict)
                start_time = end_time

        # Saving
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                ckpt_path = os.path.join("/content/drive/MyDrive/JAX-PI_1D", config.wandb.name, "ckpt")
                save_checkpoint(
                    model.state, ckpt_path, keep=config.saving.num_keep_ckpts
                )

    return model
