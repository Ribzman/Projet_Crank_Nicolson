import numpy as np #Pour la gestion de données
import matplotlib.pyplot as plt #Pour l'affichage graphique
from matplotlib.animation import FuncAnimation #Pour l'animation en temps réel 
from scipy.sparse import diags #Utile pour créer les matrices diagonnales.
from scipy.sparse.linalg import spsolve #Pour resoudre les equations avec des matrices creuses.

"""Constantes du problème"""
g = 10 #intensité des interaction entre bosons
L =  20 #Longueur de l'axe X
Nx = 500 #Nombre de valeur de x
dx = np.sqrt(1/g)*(2*L)/Nx #Pas d'espace
dt = 1/g * 0.1 #Pas de temps
r = dt/(2*dx**2) #coefficient r trouvé lors de l'établissement de l'algorithme
tmax = 1000 #itérations maximum de simulation

"""Conditions initiales"""
x = np.linspace(-L, L, Nx) #On initialise un vecteur avec des valeurs de x entre -L et L
V = (x**2)/2 #On choisit une potentiel de piègeage harmonique
psi = np.exp(0.5*(-x**2)/2).astype(complex) #On prend pour fonction d'onde de base une Gaussienne
psi_prev = psi.copy() #On copie cette fonction pour traiter la non linearité plus tard

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5)) #Créer deux sous plot pour y mettre l'animation et l'affichage de la norme aucours du temps.
line, = ax1.plot(x, np.abs(psi)**2, lw=2) #règle les données à afficher en axe X et Y
ax1.set_ylim(0, 1) #Etablit une limite pour l'axe Y
ax1.set_ylabel('|psi|**2') #Donne un nom à l'axe Y
ax1.set_xlabel('x') #Donne un nom à l'axe X

"""Créer un texte et le modifie pour faire la simulation en fonction de l'interaction entre les bosons"""
txt_g = ax1.text(0.05, 0.9, '', transform=ax1.transAxes, fontweight='bold')
txt_g.set_text(f"Simulation GPE 1D (g = {g})")

"""On règle les parmètres de l'autre graphique."""
norms = [] #liste des valeurs de la norme de Psi.
times = [] #liste des iterations ou on calcule la norme de Psi.
line2, = ax2.plot([], [], lw=2, color='red') #règle les données à afficher en axe X et Y
ax2.set_xlim(0, 100) #Règlage de la limite de l'axe X par defaut à 100.
ax2.set_ylim(0, 1) #Règlage de la limite de l'axe Y par defaut à 1.
ax2.set_title("Norme de Psi") #Titre du Graphique.

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

def animate(i): #definit la fonction d'animation et le corps d'excution de l'algorithme

    global psi
    global psi_prev

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

    current_norm = np.sum(np.abs(psi)**2 * dx) #On calcule la norme.
    """On ajoute à chaque iteration la norme et l'instant dans la liste correspondante."""
    norms.append(current_norm)
    times.append(i)

    """On redimensionne les axes si besoin."""
    if i > 100:
        ax2.set_xlim(0, 100+i)
    
    line.set_ydata(np.abs(psi)**2) #On definit la donnée à utiliser
    line2.set_data(times[-tmax:], norms[-tmax:]) 
    #On definit la donnée à utiliser (tmax premieres normes et instants)
    return line, line2 #on retourne les deux courbes.

"""On définit la fênetre d'animation à partir de la fonction puis l'affiche"""
ani = FuncAnimation(fig, animate, frames=tmax, interval=20, blit=False)
plt.tight_layout()
plt.show()
