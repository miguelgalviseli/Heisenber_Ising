# Importar Ising_class

import Ising_class as Ising
import numpy as np

L = 100 # Lado de la red
T = 1.0 # Temperatura
nsteps = 100000 # Número de pasos
J = 0 # Intercambio
H = np.linspace(-5, 5, 100) # Campo magnético
if __name__=='__main__':
    print("Veamos las gráficas pedidas:")


    ising_model = Ising.Ising2D(L, T, nsteps, J, H)
    #ising_model.plot_magnetizacion()
    #ising_model.plot_energia()
    #ising_model.plot_energia_1()
    ising_model.plot_configuraciones()






