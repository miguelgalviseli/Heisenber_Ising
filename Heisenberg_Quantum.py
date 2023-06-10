#Librerias
import numpy as np
import matplotlib.pyplot as plt
import random, math
from scipy.optimize import minimize
from numpy.random import rand
import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit
plt.rcParams['figure.figsize'] = 14, 7 #ancho, alto

font = {'weight' : 'bold',
        'size'   : 15}
random.seed(1) # semilla inicial

#Clase Ising2D que contiene los métodos para resolver el modelo de Ising en 2D
class Heisenberg2D:
    def __init__(self, n, J, Jm, h, H_ajuste,T1,T2,T3):
        self.n = n
        self.J = J
        self.Jm = Jm
        self.h = h
        self.H_ajuste = H_ajuste
        # Generar configuración inicial de momentos magnéticos
        self.Momentos = np.random.rand(self.n, self.n, 3)
        self.Momentos[:, :, 2] = np.where(self.Momentos[:, :, 2] < 0.5, -1, 1)
        self.T1=T1
        self.T2=T2
        self.T3=T3
        self.Theta_quantum = np.linspace(-np.pi, np.pi, int(2*self.J+1)).tolist()
        self.Theta_classic = np.linspace(-np.pi, np.pi, 1000).tolist()


    def vecinosp(self, i, j):

        V = []
        if i > 0:
            V.append(self.Momentos[i-1][j])
        if i < self.n-1:
            V.append(self.Momentos[i+1][j])
        if j > 0:
            V.append(self.Momentos[i][j-1])
        if j < self.n-1:
            V.append(self.Momentos[i][j+1])
        return V
    
    def Ein(self,mu,h_3):
        sum = 0
        if self.Jm != 0:
            for v in self.vecinosp():
                sum += np.dot(mu, v)
        return (-self.Jm * sum - np.dot(mu, h_3))
    
    def Info(self,h_2,T,Theta):
        H = np.array([0, 0, float(h_2)])
        k_B = 1
        Etot = 0
        Mtot = 0
        mu=self.Momentos
        for i in range(self.n):
            for j in range(self.n):
                V = self.vecinosp(i, j)
                Etot += self.Ein(mu[i][j],H)
                Mtot += self.Momentos[i][j][2]
        
        EE = []
        MM = []
        iter = 200
        Phi = np.linspace(0, 2*np.pi, 100)
        
        for r in range(iter):
            for i in range(self.n):
                for j in range(self.n):
                    mu = self.Momentos[i][j]
                    V = self.vecinosp(i, j)
                    
                    random.seed()
                    phi = np.random.choice(Phi)
                    theta = np.random.choice(Theta)
                    mutrial = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
                    
                    E_1 = self.Ein(mu,H)
                    E_2 = self.Ein(mutrial,H)
                    deltaE = E_2 - E_1
                    deltaM = mutrial[2] - mu[2]
                    
                    if deltaE <= 0:
                        self.Momentos[i][j] = mutrial
                        Etot += deltaE
                        Mtot += deltaM
                    else:
                        rand = np.random.random()
                        W = np.exp(-(E_2 - E_1) / (k_B * T))
                        if rand <= W:
                            self.Momentos[i][j] = mutrial
                            Etot += deltaE
                            Mtot += deltaM
            
            EE.append(float(Etot))
            MM.append(float(Mtot))
        
        return np.arange(iter), EE, MM, np.mean(self.Momentos[:, :, 2]), self.Momentos
    
    def mag_isotermas_quantum(self):

        T3=[self.T3]
        T2=[self.T2]
        T1=[self.T1]
        

        for t in tqdm(T1):
            Mo1=[]
            for s in self.h:
                F1=self.Info(s,t,self.Theta_quantum)
                Mo1.append(float(F1[3]))  
                
        for t in tqdm(T2):
            Mo2=[]
            for s in self.h:
                F2=self.Info(s,t,self.Theta_quantum)
                Mo2.append(float(F2[3]))  
                

        for t in tqdm(T3):
            Mo3=[]
            for s in self.h:
                F3=self.Info(s,t,self.Theta_quantum)
                Mo3.append(float(F3[3]))  


        #Ajustemos la función de Brillouin a los datos y sciPy nos da los parámetros de ajuste
        #Escribamos B con la tangente hiperbólica como un polinomio de grado 7 de (H/T

        def B(H,T,a,b,c,d,e,f,g): #función de Brillouin para ajustar con polinomio de grado 12 de (H/T)

            return a*(H/T)**11 + b*(H/T)**9 + c*(H/T)**7 + d*(H/T)**5 + e*(H/T)**3 + f*(H/T) + g

        #Ajustemos la función de Brillouin a los datos y sciPy nos da los parámetros de ajuste
        popt1, pcov1 = curve_fit(B, self.h, Mo1)
        popt2, pcov2 = curve_fit(B, self.h, Mo2)
        popt3, pcov3 = curve_fit(B, self.h, Mo3)
            
        #Grafiquemos los datos y el ajuste
        
        plt.figure(figsize=(12,8))
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        plt.title('Magnetización en función de campo magnético, Modelo de Heisenberg cuántico',fontsize=20)
        plt.plot(self.h,Mo1,'o',label='T={}K'.format(T1[0]))
        plt.plot(self.h,Mo2,'o',label='T={}K'.format(T2[0]))
        plt.plot(self.h,Mo3,'o',label='T={}K'.format(T3[0]))
        plt.plot(self.H_ajuste,B(self.H_ajuste,*popt1),'--',label='Ajuste T={}K'.format(T1[0]))
        plt.plot(self.H_ajuste,B(self.H_ajuste,*popt2),'--',label='Ajuste T={}K'.format(T2[0]))
        plt.plot(self.H_ajuste,B(self.H_ajuste,*popt3),'--',label='Ajuste T={}K'.format(T3[0]))
        plt.xlabel('Campo magnético',fontsize= 20)
        plt.ylabel('Magnetización',fontsize= 20)
        plt.legend(fontsize= 15)
        plt.show()
    #Veamos el modelo clásico
    def mag_isotermas_classic(self):

        T3=[self.T3]
        T2=[self.T2]
        T1=[self.T1]
        

        for t in tqdm(T1):
            Mo11=[]
            for s in self.h:
                F1=self.Info(s,t,self.Theta_classic)
                Mo11.append(float(F1[3]))  
                
        for t in tqdm(T2):
            Mo12=[]
            for s in self.h:
                F2=self.Info(s,t,self.Theta_classic)
                Mo12.append(float(F2[3]))  
                

        for t in tqdm(T3):
            Mo13=[]
            for s in self.h:
                F3=self.Info(s,t,self.Theta_classic)
                Mo13.append(float(F3[3]))  


        #GRafiquemos la magnetización evolucionando con las iteraciones
        itera = self.Info(self.h[0],self.T1,self.Theta_classic)
        plt.figure(figsize=(12, 4))
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.plot(itera[0], itera[2], linewidth=2, color='blue')
        plt.xlabel('Iteraciones', fontsize=15)
        plt.ylabel('Magnetización', fontsize=15)
        plt.title('Evolución de la Magnetización', fontsize=20)

        plt.tight_layout()
        plt.show()

        def L(H,T,a): #función de Langevin para ajustar
            return a*np.tanh((H/T)) 

        #Ajustemos la función de Langevin a los datos y sciPy nos da los parámetros de ajuste
        popt11, pcov11 = curve_fit(L, self.h, Mo11)
        popt12, pcov12 = curve_fit(L, self.h, Mo12)
        popt13, pcov13 = curve_fit(L, self.h, Mo13)
            
        #Grafiquemos los datos y el ajuste
        plt.figure(figsize=(12,8))
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        plt.title('Magnetización en función de campo magnético, Modelo de Heisenberg clásico',fontsize=20)
        plt.plot(self.h,Mo11,'o',label='T={}K'.format(T1[0]))
        plt.plot(self.h,Mo12,'o',label='T={}K'.format(T2[0]))
        plt.plot(self.h,Mo13,'o',label='T={}K'.format(T3[0]))
        plt.plot(self.H_ajuste,L(self.H_ajuste,*popt11),'--',label='Ajuste T={}K'.format(T1[0]))
        plt.plot(self.H_ajuste,L(self.H_ajuste,*popt12),'--',label='Ajuste T={}K'.format(T2[0]))
        plt.plot(self.H_ajuste,L(self.H_ajuste,*popt13),'--',label='Ajuste T={}K'.format(T3[0]))
        plt.xlabel('Campo magnético',fontsize= 20)
        plt.ylabel('Magnetización',fontsize= 20)
        plt.legend(fontsize= 15)
        plt.show()