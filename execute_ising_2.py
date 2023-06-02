import Ising_class_2 as ising
import numpy as np

if __name__ == "__main__":
    L = 100  # Lado de la red
    T = 2.0  # Temperatura
    nsteps = 40000  # Número de pasos
    J = 1  # Intercambio
    H = np.linspace(-25, 25, 100)  # Campo magnético
    param = [1, 1]

    # ajuste de la curva de magnetización en función del campo magnético
    ejemplo = ising.Ising_v2(L, T, nsteps, J, H, param)
    print(ejemplo.ajuste())