import numpy as np
from scipy.optimize import minimize
from Ising_class import Ising2D
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm


class IsingV2:
    def __init__(self, L, T, nsteps, J, H, param):
        self.L = L
        self.T = T
        self.nsteps = nsteps
        self.J = J
        self.H = H
        self.param = param
        self.iso = Ising2D(self.L, self.T, self.nsteps, self.J, self.H).ising2D()[6]
        self.iso2 = Ising2D(self.L, self.T, self.nsteps, self.J, np.flip(self.H)).ising2D()[6]

    def tanh(self, x, a, b, c):
        return a * np.tanh(b * x + c)

    # Ajustemos la curva de la isotermas para T1 T2 y T3 y grafiquemos con subplots
    def ajuste(self):
        popt, pcov = curve_fit(self.tanh, self.H, self.iso)

        # graficar ajuste
        plt.figure(figsize=(10, 8))
        plt.plot(self.H, self.iso, label="T={}K".format(self.T), color="red")
        plt.plot(self.H, self.tanh(self.H, *popt), '*', label="Ajuste", color="black")
        plt.grid()
        plt.legend()
        plt.show()

        # graficar histéresis
        plt.figure(figsize=(10, 8))
        plt.title("Histeresis para un ferromagneto en el modelo de ising en 2D", fontsize=20)
        plt.plot(np.arange(self.H[0], self.H[-1], 0.5), self.iso, label="T={}K".format(self.T), color="green")
        plt.plot(np.arange(self.H[-1], self.H[0], -0.5), self.iso2, label="T={}K".format(self.T), color="black")
        plt.xlabel("H", fontsize=20)
        plt.ylabel("M", fontsize=20)
        plt.xlim(-5, 5)
        plt.legend()
        plt.show()

        return


class IsingTemp(IsingV2):
    def __init__(self, L, T, nsteps, J, H, param, N, nt, eq_steps, mc_steps):
        super().__init__(self, L, T, nsteps, J, H)

        self.N = N  # lado de la red
        self.nt = nt  # número de puntos de temperatura
        self.eq_steps = eq_steps  # número de pasos de equilibrio
        self.mc_steps = mc_steps  # número de pasos de Monte Carlo

    def initial_state(self):
        r""" Genera un estado aleatorio para una red NxN"""
        state = 2 * np.random.randint(2, size=(self.N, self.N)) - 1
        return state

    def mcmove(self, config, beta):
        r"""Movimiento de Monte Carlo usando el algoritmo de Metropolis"""

        for i in range(self.N):
            for j in range(self.N):
                a = np.random.randint(0, self.N)
                b = np.random.randint(0, self.N)
                s = config[a, b]
                nb = config[(a + 1) % self.N, b] + config[a, (b + 1) % self.N] + config[(a - 1) % self.N, b] + \
                     config[
                         a, (b - 1) % self.N]
                cost = 2 * s * nb

                if cost < 0:
                    s *= -1
                elif np.random.rand() < np.exp(-cost * beta):
                    s *= -1
                config[a, b] = s
        return config

    def calc_energy(self, config):
        r"""Energía de una configuración dada."""

        energy = 0

        for i in range(len(config)):
            for j in range(len(config)):
                S = config[i, j]
                nb = config[(i + 1) % self.N, j] + config[i, (j + 1) % self.N] + config[(i - 1) % self.N, j] + \
                     config[
                         i, (j - 1) % self.N]
                energy += -nb * S
        return energy / 2.  # to compensate for over-counting

    def calc_mag(self, config):
        r"""Magnetización de una configuración dada"""

        mag = np.sum(config)
        return mag

        # función para hacer el ciclo sobre las temperaturas

    def ct(self):
        E, M, C, X = np.zeros(self.nt), np.zeros(self.nt), np.zeros(self.nt), np.zeros(self.nt)
        n1, n2 = 1.0 / (self.mc_steps * self.N * self.N), 1.0 / (self.mc_steps * self.mc_steps * self.N * self.N)

        for tt in tqdm(range(self.nt)):
            config = self.initial_state()  # generar una configuración aleatoria
            E1 = M1 = E2 = M2 = 0
            iT = 1.0 / self.T[tt]
            iT2 = iT * iT

            for i in range(self.eq_steps):
                self.mcmove(config, iT)
                ene = self.calc_energy(config)
                mag = self.calc_mag(config)

                E1 = E1 + ene
                M1 = M1 + mag
                M2 = M2 + mag * mag
                E2 = E2 + ene * ene

            # guardar los valores de energía y magnetización
            E[tt] = n1 * E1
            M[tt] = n1 * M1
            C[tt] = (n1 * E2 - n2 * E1 * E1) * iT2

        return E, M, C

    def graficar(self):
        plt.figure(figsize=(10, 8))
        plt.plot(np.arange(0, self.T, 100), self.ct()[0], 'o')
        plt.show()

        return
