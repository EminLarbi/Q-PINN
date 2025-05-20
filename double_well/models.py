from functools import partial

import jax.numpy as jnp
from jax import lax, jit, grad, vmap
import jax.debug as jaxdebug
from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from matplotlib import pyplot as plt
from math import sqrt


class QPINN(ForwardIVP):

    def __init__(
        self,
        config,
        psi_real0,
        psi_imag0,
        t_star,
        x_star,
        hbar,
        mass,
        e,
        k,
        x0,
        Tcar,
        Xcar,
        Mcar,
    ):
        super().__init__(config)

        self.psi_real0 = psi_real0
        self.psi_imag0 = psi_imag0
        self.t_star = t_star
        self.x_star = x_star
        self.e = e
        self.k = k
        self.hbar = hbar
        self.mass = mass

        self.Tcar = Tcar
        self.Xcar = Xcar
        self.Mcar = Mcar

        self.x0 = x0

        self.t0 = t_star[0]
        self.t1 = t_star[-1]

        # Predictions over a grid
        self.real_pred_fn = vmap(vmap(self.real_net, (None, None, 0)), (None, 0, None))
        self.imag_pred_fn = vmap(vmap(self.imag_net, (None, None, 0)), (None, 0, None))

        self.r_pred_fn = vmap(vmap(self.r_net, (None, None, 0)), (None, 0, None))

    def real_net(self, params, t, x):
        z = jnp.stack([t, x])
        outs = self.state.apply_fn(params, z)
        psi_real = outs[0]
        return psi_real

    def imag_net(self, params, t, x):
        z = jnp.stack([t, x])
        outs = self.state.apply_fn(params, z)
        psi_imag = outs[1]
        return psi_imag

    def r_net(self, params, t, x):
        # Obtener las partes reales e imaginarias de la función de onda
        psi_real = self.real_net(params, t, x)
        psi_imag = self.imag_net(params, t, x)

        # Derivadas temporales
        real_t = grad(self.real_net, argnums=1)(params, t, x)
        imag_t = grad(self.imag_net, argnums=1)(params, t, x)

        # Derivadas espaciales segundas
        real_xx = grad(grad(self.real_net, argnums=2), argnums=2)(params, t, x)
        imag_xx = grad(grad(self.imag_net, argnums=2), argnums=2)(params, t, x)

        # Evaluar el potencial para el valor de x actual
        V = self.V(x)

        # Aplicar el operador Hamiltoniano con las constantes físicas
        H_psi_real = -0.5 * (self.hbar**2 / self.mass) * real_xx + V * psi_real
        H_psi_imag = -0.5 * (self.hbar**2 / self.mass) * imag_xx + V * psi_imag

        # Residuales de las ecuaciones reales e imaginarias
        residual_real = -self.hbar * imag_t - H_psi_real
        residual_imag = self.hbar * real_t - H_psi_imag

        return residual_real, residual_imag

    def V(self, x):
        # Constante de energía base
        V0 = 8 * self.Tcar**2 / (self.Mcar * self.Xcar**2)
        # Posición de los pozos (en unidades adimensionales x/Xcar)
        a = self.x0
        # Altura de barrera garantizada V0
        barrier_energy = V0
        print(f"Energía barrera: {barrier_energy}")

        # Potencial doble pozo calculado para todo x (vectorizado)
        potential = V0 * (x**2 - a**2)**2 / a**4

        # Definir ancho de la zona alrededor de cada mínimo donde V = 0
        delta = 0.5 * jnp.abs(a)  # 5% de |a| (ajustable)

        # Máscara booleana con jnp.where (vectorizada)
        mask = (jnp.abs(x - a) <= delta) | (jnp.abs(x + a) <= delta)
        return jnp.where(mask, 0.0, potential)

    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch):
        # Sort temporal coordinates
        t_sorted = batch[:, 0].sort()
        real_pred, imag_pred = self.r_pred_fn(params, t_sorted, batch[:, 1])

        real_pred = real_pred.reshape(self.num_chunks, -1)
        imag_pred = imag_pred.reshape(self.num_chunks, -1)

        real_l = jnp.mean(real_pred**2, axis=1)
        imag_l = jnp.mean(imag_pred**2, axis=1)

        real_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ real_l)))
        imag_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ imag_l)))

        # Take minimum of the causal weights
        gamma = jnp.vstack([real_gamma, imag_gamma])
        gamma = gamma.min(0)

        return real_l, imag_l, gamma

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Initial condition loss
        real0_pred = vmap(self.real_net, (None, None, 0))(params, self.t0, self.x_star)
        imag0_pred = vmap(self.imag_net, (None, None, 0))(params, self.t0, self.x_star)

        real0_loss = jnp.mean((real0_pred - self.psi_real0) ** 2)
        imag0_loss = jnp.mean((imag0_pred - self.psi_imag0) ** 2)

        # Residual loss
        if self.config.weighting.use_causal == True:
            real_l, imag_l, gamma = self.res_and_w(params, batch)
            real_loss = jnp.mean(real_l * gamma)
            imag_loss = jnp.mean(imag_l * gamma)

        else:
            real_pred, imag_pred = self.r_pred_fn(params, batch[:, 0], batch[:, 1])
            # Compute loss
            real_loss = jnp.mean(real_pred**2)
            imag_loss = jnp.mean(imag_pred**2)

        loss_dict = {
            "real_ic": real0_loss,
            "imag_ic": imag0_loss,
            "real_re": real_loss,
            "imag_re": imag_loss,
        }
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        real0_ntk = vmap(ntk_fn, (None, None, None, 0))(
            self.real_net, params, self.t0, self.x_star
        )

        imag0_ntk = vmap(ntk_fn, (None, None, None, 0))(
            self.imag_net, params, self.t0, self.x_star
        )

        # Consider the effect of causal weights
        if self.config.weighting.use_causal:
            batch = jnp.array([batch[:, 0].sort(), batch[:, 1]]).T
            real_loss_ntk = vmap(ntk_fn, (None, None, 0, 0))(
                self.real_net, params, batch[:, 0], batch[:, 1]
            )
            imag_loss_ntk = vmap(ntk_fn, (None, None, 0, 0))(
                self.imag_net, params, batch[:, 0], batch[:, 1]
            )

            real_loss_ntk = real_loss_ntk.reshape(self.num_chunks, -1)
            imag_loss_ntk = imag_loss_ntk.reshape(self.num_chunks, -1)

            real_loss_ntk = jnp.mean(real_loss_ntk, axis=1)
            imag_loss_ntk = jnp.mean(imag_loss_ntk, axis=1)

            _, _, casual_weights = self.res_and_w(params, batch)
            real_loss_ntk = real_loss_ntk * casual_weights
            imag_loss_ntk = imag_loss_ntk * casual_weights
        else:
            real_loss_ntk = vmap(ntk_fn, (None, None, 0, 0))(
                self.real_net, params, batch[:, 0], batch[:, 1]
            )
            imag_loss_ntk = vmap(ntk_fn, (None, None, 0, 0))(
                self.imag_net, params, batch[:, 0], batch[:, 1]
            )

        ntk_dict = {
            "real_ic": real0_ntk,
            "imag_ic": imag0_ntk,
            "real_re": real_loss_ntk,
            "imag_re": imag_loss_ntk,
        }
        return ntk_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, real_test, imag_test):
        real_pred = self.real_pred_fn(params, self.t_star, self.x_star)
        imag_pred = self.imag_pred_fn(params, self.t_star, self.x_star)
        real_error = jnp.linalg.norm(real_pred - real_test) / jnp.linalg.norm(real_test)
        imag_error = jnp.linalg.norm(imag_pred - imag_test) / jnp.linalg.norm(imag_test)
        return real_error, imag_error


class QPINNEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, psi_real, psi_imag):
        real_error, imag_error = self.model.compute_l2_error(
            params,
            psi_real,
            psi_imag,
        )
        self.log_dict["real_error"] = real_error
        self.log_dict["imag_error"] = imag_error

    def log_preds(self, params):
        real_pred = self.model.real_pred_fn(
            params, self.model.t_star, self.model.x_star
        )
        fig = plt.figure(figsize=(6, 5))
        plt.imshow(real_pred.T, cmap="jet")
        self.log_dict["real_pred"] = fig
        plt.close()
        imag_pred = self.model.imag_pred_fn(
            params, self.model.t_star, self.model.x_star
        )
        fig = plt.figure(figsize=(6, 5))
        plt.imshow(imag_pred.T, cmap="jet")
        self.log_dict["imag_pred"] = fig
        plt.close()

    def __call__(self, state, batch, psi_real, psi_imag):
        self.log_dict = super().__call__(state, batch)

        if self.config.weighting.use_causal:
            _, _, causal_weight = self.model.res_and_w(state.params, batch)
            self.log_dict["cas_weight"] = causal_weight.min()

        if self.config.logging.log_errors:
            self.log_errors(state.params, psi_real, psi_imag)

        if self.config.logging.log_preds:
            self.log_preds(state.params)

        return self.log_dict
