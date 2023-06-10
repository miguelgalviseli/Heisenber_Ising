# Importar Heisenberg_class

import Heisenberg_Quantum as Heisenberg
import numpy as np
J=3/2 #Momento angular total
Jm=0 #Intercambio de interacción magnética
h=np.arange(-100,110,5) #campo magnetico para las isotermas
hh=np.linspace(-100,100,1000)  #campo magnetico para el ajuste
n=10 #Lado de la red
#Temperaturas para el caso cuántico
T1=10
T2=40
T3=70
#Temperaturas para el caso clásico
T4=2
T5=10
T6=50

if __name__=='__main__':
    print("Veamos las gráficas pedidas pero tenga un poco de paciencia, la presente configuración tarda un poco, al rededor de 15 minutos:")


    Heisenberg_model1 = Heisenberg.Heisenberg2D(n, J, Jm, h, hh,T1,T2,T3)
    Heisenberg_model1.mag_isotermas_quantum()
    Heisenberg_model2 = Heisenberg.Heisenberg2D(n, J, Jm, h, hh,T4,T5,T6)
    Heisenberg_model2.mag_isotermas_classic()