import Ising_class_2 as ising
import numpy as np

if __name__ == "__main__":
    L = 100  # Lado de la red
    T = 3.0  # Temperatura
    nsteps = 40000  # Número de pasos
    J = 1  # Intercambio
    H = np.linspace(-25, 25, 100)  # Campo magnético
    param = [1, 1]

    # ajuste de la curva de magnetización en función del campo magnético
    #ejemplo = ising.IsingV2(L, T, nsteps, J, H, param)
    #print(ejemplo.ajuste())

    # dependiente de la temperatura
    N = 10
    nt = 32
    eq_steps = 2**8
    mc_steps = 2**10

    ejemplo2 = ising.IsingTemp(L, T, nsteps, J, H, param, N, nt, eq_steps, mc_steps)