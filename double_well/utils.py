import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.signal import find_peaks
import plotly.graph_objects as go
from math import sqrt




   

def gaussian_wave_packet(x, t, x0, sigma, k0, A, hbar, m):
    i = 1j
    z = 1 + i * (hbar * t) / (2 * m * sigma**2)
    phase = k0 * x - (hbar * k0**2 * t) / (2 * m)
    exponent = -((x - x0 - (hbar * k0 * t) / m) ** 2) / (4 * sigma**2 * z)
    psi = (A / jnp.sqrt(z)) * jnp.exp(exponent) * jnp.exp(i * phase)
    return psi




def compute_energy(psi, x, hbar, m):
    """
    Calcula la energía cinética promedio ⟨ψ|T|ψ⟩ con T = -(ħ²/2m) d²/dx².
    Parámetros:
      psi   : array complejo, ψ(x)
      x     : array real, posiciones x
      hbar  : float, constante de Planck reducida
      m     : float, masa de la partícula
    Retorna:
      E : float, energía esperada
    """
    dx = x[1] - x[0]
    # Segunda derivada mediante diferencias finitas
    d2psi = np.gradient(np.gradient(psi, dx), dx)
    # Valor esperado de la energía cinética
    integrand = np.conj(psi) * ( - (hbar**2) / (2 * m) * d2psi )
    E = np.real(np.trapz(integrand, x))
    return E


def get_dataset(
    Tmin=0,
    Tmax=2.0,
    Lmin=-10,
    Lmax=10,
    n_t=200,
    n_x=128,
    x0=0.0,
    sigma=1.0,
    k0=1.0,
    A=1.0,
    hbar=1.0,
    m=1.0,
    e=1,
    k=1,
    Tcar=1,
    Xcar=1,
):
    t_star = jnp.linspace(Tmin, Tmax, int(n_t))
    x_star = jnp.linspace(Lmin, Lmax, int(n_x))
    psi_t_x = vmap(
        vmap(
            lambda t, x: gaussian_wave_packet(x, t, x0, sigma, k0, A, hbar, m),
            (None, 0),
        ),
        (0, None),
    )(t_star, x_star)
    psi_real = jnp.real(psi_t_x)
    psi_imag = jnp.imag(psi_t_x)
    # Onda inicial completa (numpy)
    psi_initial = np.array(psi_real[0, :]) + 1j * np.array(psi_imag[0, :])
    x_np = np.array(x_star)

    # Cálculo de la energía inicial
    energia_inicial = compute_energy(psi_initial, x_np, hbar, m)
    print(f"Energía inicial: {energia_inicial}")

    return (
        psi_real,
        psi_imag,
        t_star,
        x_star,
        Tcar,
        Xcar,
        np.array(psi_real[0, :]),
        np.array(psi_imag[0, :]),
    )



