import numpy as np #Pour la gestion de données
import matplotlib.pyplot as plt #Pour l'affichage graphique
from matplotlib.animation import FuncAnimation #Pour l'animation en temps réel 
from scipy.sparse import diags #Utile pour créer les matrices diagonnales.
from scipy.sparse.linalg import spsolve #Pour resoudre les equations avec des matrices creuses.

"""Constantes du problème"""
g = 100 #intensité des interaction entre bosons
L =  20 #Longueur de l'axe X
Nx =    1024 #Nombre de valeur de x
dx = np.sqrt(1/g)*(2*L)/2*Nx #Pas d'espace
#dt = 1/g * 0.1
dt = 0.25 * dx**2  #Pas de temps
r = dt/(2*dx**2) #coefficient r trouvé lors de l'établissement de l'algorithme
tmax = 5000 #itérations maximum de simulation

"""Conditions initiales"""
x = np.linspace(-L, L, Nx) #On initialise un vecteur avec des valeurs de x entre -L et L
V = (x**2)/2 #On choisit une potentiel de piègeage harmonique
psi0 = 1.0
xi = 1.0 / np.sqrt(g)
psi = (psi0 * np.tanh(x / (np.sqrt(2) * xi))).astype(complex)
psi_prev = psi.copy()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 5)) #Créer deux sous plot pour y mettre l'animation et l'affichage de la norme aucours du temps.
line, = ax1.plot(x, np.abs(psi)**2, lw=2) #règle les données à afficher en axe X et Y
ax1.set_ylim(0, 1) #Etablit une limite pour l'axe Y
ax1.set_ylabel(r'$|\Psi|²$') #Donne un nom à l'axe Y
ax1.set_xlabel('Position(m)') #Donne un nom à l'axe X

"""Créer un texte et le modifie pour faire la simulation en fonction de l'interaction entre les bosons"""
ax1.set_title(f"Simulation GPE 1D (g = {g})")
g
"""On règle les parmètres de l'autre graphique."""
norms = [] #liste des valeurs de la norme de Psi.
times = [] #liste des iterations ou on calcule la norme de Psi.
line2, = ax2.plot([], [], lw=2, color='red') #règle les données à afficher en axe X et Y
#ax2.set_xlim(0, 0.05) #Règlage de la limite de l'axe X par defaut à 100.
ax2.set_ylim(0, 1) #Règlage de la limite de l'axe Y par defaut à 1.
ax2.set_title(r"Norme de $\Psi$") #Titre du Graphique.
ax2.set_xlabel("Temps(s)") #titre de l'axe x
ax2.set_ylabel(r"Norme de $\Psi$") #titre de l'axe y

energies = [] #liste des énergies calculées
line3, = ax3.plot([],[], lw=2, color ='green')
#ax3.set_xlim(0, 0.05) #Règlage de la limite de l'axe X par defaut à 100.
#ax3.set_ylim(0, 1) #Règlage de la limite de l'axe Y par defaut à 1.
ax3.set_title(r"Energie de $\Psi$") #Titre du Graphique.
ax3.set_xlabel("Temps(s)") #titre de l'axe x
ax3.set_ylabel(r"Energie de $\Psi$") #titre de l'axe y

"""On definit les diagonales de A (Inferieure, Principale et Superieure) 
puis on construit la matrice tridiagonales au format Compressed Sparse Column 
pour plus de rapidité de calcul"""
Principale_DiagA = (1 + 1j*r + 0.5j*dt*V) * np.ones(Nx)
Diags_Inf_SuppA = -0.5j*r * np.ones(Nx-1)
A = diags([Diags_Inf_SuppA, Principale_DiagA, Diags_Inf_SuppA], [1,0,-1]).tocsc()

"""On definit les diagonales de B (Inferieure, Principale et Superieure) 
puis on construit la matrice tridiagonales au format Compressed Sparse Column 
pour plus de rapidité de calcul"""
Principale_DiagB = (1 - 1j*r - 0.5j*dt*V) * np.ones(Nx)
Diags_Inf_SuppB = 0.5j*r * np.ones(Nx-1)
B = diags([Diags_Inf_SuppB, Principale_DiagB, Diags_Inf_SuppB], [1,0,-1]).tocsc()

"""Fonction calculant la norme de Psi"""
def calculate_norm(psi):
    current_norm = np.sum(np.abs(psi)**2 * dx)
    return current_norm

"""Fonction calculant l'energie du condensat"""
def calculate_energy(psi, psi_prev):
    psi_extrap = 1.5 * psi - 0.5 * psi_prev #Calcul de Psi extrapolée
    extrap_density = np.abs(1.5 * psi - 0.5 * psi_prev)**2  #Calcul de la densité extrapolée
    
    grad2_psi = np.gradient(np.gradient(psi_extrap, dx), dx) #Calcul du gradient 2 ème de psi
    grad_density = -np.real(np.conj(psi_extrap) * grad2_psi) #Calcul du terme en densité gradient 

    current_energy = np.sum((1/2 * grad_density + V * extrap_density + g/2 * extrap_density**2)* dx) #Calcul de l'énergie
    return current_energy

steps_per_frame = 50
phys_time = 0

def animate(i): #definit la fonction d'animation et le corps d'excution de l'algorithme

    global psi, psi_prev, phys_time

    for _ in range(0, steps_per_frame):
        extrap_density = np.abs((1.5 * psi - 0.5 * psi_prev))**2 
        #Calcule la densité a partir de l'extrapolation.

        """Calcule la matrice diagonale et la convertit 
        en Compressed Sparse Column pour plus de rapidité de calcul"""
        Principale_DiagO = 0.5j*dt*g*extrap_density * np.ones(Nx)
        O = diags([Principale_DiagO],[0]).tocsc() 
    

        Aprime = A + O #Calcule la matrice A prime
        Bprime = B - O #Calcule la metrice B prime
        res = Bprime.dot(psi) #On multiplie par la fonction psi
        
        psi_next = spsolve(Aprime, res) #On resoud pour l'itération suivante de psi

        """On met a jour psi_prev et psi"""
        psi_prev = psi.copy() 
        psi = psi_next.copy()

        phys_time += dt

    """On ajoute à chaque iteration la norme, l'energie et l'instant dans la liste correspondante."""
        
    norms.append(calculate_norm(psi))
    energies.append(calculate_energy(psi, psi_prev))
    times.append(phys_time)

    """On redimensionne les axes si besoin."""
    if len(times) > 1:
        ax2.set_xlim(0, times[-1])
        ax3.set_xlim(0, times[-1])

    if len(norms) > 1:
        ax2.set_ylim(min(norms) * 0.9, max(norms) * 1.1) 

    if len(energies) > 1:
        ax3.set_ylim(min(energies) * 0.9, max(energies) * 1.1)
    
    line.set_ydata(np.abs(psi)**2) #On definit la donnée à utiliser
    line2.set_data(times[-tmax:], norms[-tmax:]) 
    line3.set_data(times[-tmax:], energies[-tmax:])
    #On definit la donnée à utiliser (tmax premieres normes, énergies et instants)
    return line, line2, line3 #on retourne les deux courbes.

"""On définit la fênetre d'animation à partir de la fonction puis l'affiche"""
ani = FuncAnimation(fig, animate, frames=500, interval=20, blit=False)
plt.tight_layout()
plt.show()