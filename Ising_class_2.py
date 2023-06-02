import numpy as np
from scipy.optimize import minimize
from Ising_class import Ising2D
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class Ising_v2():
    def __init__(self, L, T, nsteps, J, H, param):
        self.L = L
        self.T = T
        self.nsteps = nsteps
        self.J = J
        self.H = H
        self.param = param
        self.iso = Ising2D(self.L, self.T, self.nsteps, self.J, self.H).ising2D()[6]

    def tanh(self, x,a,b,c):
        return a*np.tanh(b*x+c)
    
    #Ajustemos la curva de la isotermas para T1 T2 y T3 y grafiquemos con subplots
    def ajuste(self):
        popt, pcov = curve_fit(self.tanh, self.H, self.iso)

        plt.figure(figsize=(10, 8))
        plt.plot(self.H, self.iso,label="T={}K".format(self.T),color="red")
        plt.plot(self.H, self.tanh(self.H, *popt), '*' , label="Ajuste",color="black")
        plt.grid()
        plt.legend()
        plt.show()

        return