import random
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class Ising2D:
    def __init__(self, L, T, nsteps, J, H):
        self.L = L
        self.T = T
        self.nsteps = nsteps
        self.J = J
        self.H = H
        self.energy = 0
        self.N = L * L
        self.beta = 1.0 / T
        self.nbr = {i: ((i // L) * L + (i + 1) % L, (i + L) % self.N,
                        (i // L) * L + (i - 1) % L, (i - L) % self.N) \
                    for i in range(self.N)}
        self.S = [random.choice([1, -1]) for k in range(self.N)]
        self.R = [random.choice([1, -1]) for k in range(self.N)]

    def x_y(self, k):
        y = k // self.L
        x = k - y * self.L
        return x, y

    def ising2D(self):
        energy = 0
        N = self.L * self.L
        beta = 1.0 / self.T
        nbr = {i: ((i // self.L) * self.L + (i + 1) % self.L, (i + self.L) % N,
                   (i // self.L) * self.L + (i - 1) % self.L, (i - self.L) % N) \
               for i in range(N)}
        S = self.S.copy()
        R = self.R.copy()

        for k in range(N):
            energy += S[k] * sum(S[nn] for nn in nbr[k])
        energy *= 0.5

        E = []
        E_1 = []
        Mag = []
        itera = []

        for i in tqdm(range(len(self.H))):
            #Quiero guardar una copia de los snapshoots en una parte intermedia
            if i==int(len(self.H)*0.475):
                Intermedio=S.copy()
            for step in range(self.nsteps):
                k = random.randint(0, N - 1)
                delta_E = 2.0 * self.J * S[k] * sum(S[nn] for nn in nbr[k]) + self.H[i] * S[k]
                if random.uniform(0.0, 1.0) < math.exp(-beta * delta_E):
                    S[k] *= -1
                    energy += delta_E
                else:
                    S[k] = S[k]
                E_1.append(energy)
                itera.append(step)
            E.append(energy)

            M = np.sum(S) / N
            Mag.append(M)
        conf=[[0 for x in range (self.L)] for y in range (self.L)]#Configuración inicial
        conf1=[[0 for x in range (self.L)] for y in range (self.L)]#Configuración inicial
        conf_int=[[0 for x in range (self.L)] for y in range (self.L)]#Configuración intermedia
        for k in range(N):
            x, y = self.x_y(k)
            conf[x][y] = S[k]
        for k in range(N):
            X, Y = self.x_y(k)
            conf1[X][Y] = R[k]
        for k in range (N):
            x1,y1=self.x_y(k)
            conf_int[x1][y1]=Intermedio[k]

        return R, conf1, S, conf, sum(E) / float(len(E) * N), E, Mag, E_1, itera, conf_int

