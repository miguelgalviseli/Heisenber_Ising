# Importar Ising_class

import Ising_class as Ising
import numpy as np

if __name__ == '__main__':
    L = 100  # Lado de la red
    T = 2.0  # Temperatura
    nsteps = 4000  # Número de pasos
    J = 1  # Intercambio
    H = np.linspace(-25, 25, 100)  # Campo magnético

    print("Veamos las gráficas pedidas:")
    #ising_model = Ising.Ising2D(L, T, nsteps, J, H)

    #ising_model.plot_magnetizacion()
    #ising_model.plot_energia()
    #ising_model.plot_energia_itera()
    #ising_model.plot_ajuste()
    #ising_model.plot_histeresis()
    #ising_model.plot_configuraciones()

    nt = 32
    eq_steps = 2 ** 8
    mc_steps = 2 ** 10
    t_array = np.linspace(1., 4., nt)

    print('Gráficas Dependientes de la Temperatura')
    ejemplo = Ising.IsingTemp(L, T, nsteps, J, H, nt, eq_steps, mc_steps, t_array)
    ejemplo.ciclo_t()