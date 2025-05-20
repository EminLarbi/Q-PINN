import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt


def gaussian_wave_packet(x, t, x0, sigma, k0, A, hbar, m):
    i = 1j
    z = 1 + i * (hbar * t) / (2 * m * sigma**2)
    phase = k0 * x - (hbar * k0**2 * t) / (2 * m)
    exponent = -((x - x0 - (hbar * k0 * t) / m) ** 2) / (4 * sigma**2 * z)
    psi = (A / jnp.sqrt(z)) * jnp.exp(exponent) * jnp.exp(i * phase)
    return psi


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
    Tcar=1,
    Xcar=1,
):

    t_star = jnp.linspace(Tmin, Tmax, n_t)
    x_star = jnp.linspace(Lmin, Lmax, n_x)

    psi_t_x = vmap(
        vmap(
            lambda t, x: gaussian_wave_packet(x, t, x0, sigma, k0, A, hbar, m),
            (None, 0),
        ),
        (0, None),
    )(t_star, x_star)
    psi_real = jnp.real(psi_t_x)
    psi_imag = jnp.imag(psi_t_x)
    t_star = t_star
    x_star = x_star
    return psi_real, psi_imag, t_star, x_star, Tcar, Xcar


def main():
    # Parámetros del paquete de ondas
    T = 10.0
    L = 10
    n_t = 200
    n_x = 128
    x0 = 0.0
    sigma = 1.0
    k0 = 0.0
    A = 1.0
    hbar = 1.0
    m = 1.0

    # Calcular la función de onda
    psi_real, psi_imag, t_star, x_star = get_dataset(
        T, L, n_t, n_x, x0, sigma, k0, A, hbar, m
    )

    # Graficar la parte real e imaginaria en diferentes instantes de tiempo
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    time_indices = [0, n_t // 3, 2 * n_t // 3, n_t - 1]
    for ax, idx in zip(axs.flat, time_indices):
        ax.plot(x_star, psi_real[idx, :], label="Parte Real")
        ax.plot(x_star, psi_imag[idx, :], label="Parte Imaginaria")
        ax.set_title(f"t = {t_star[idx]:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("Amplitud")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
