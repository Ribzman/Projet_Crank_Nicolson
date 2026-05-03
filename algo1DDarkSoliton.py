import numpy as np #Pour la gestion de données
import matplotlib.pyplot as plt #Pour l'affichage graphique
from matplotlib.animation import FuncAnimation #Pour l'animation en temps réel 
from scipy.sparse import diags #Utile pour créer les matrices diagonnales.
from scipy.sparse.linalg import spsolve #Pour resoudre les equations avec des matrices creuses.

"""Constantes du problème"""
g = 1 #intensité des interaction entre bosons
L = 20 #Longueur de l'axe X
Nx = 2000 #Nombre de valeur de x
dx = np.sqrt(1/g)*(2*L)/Nx #Pas d'espace (Non modifié selon ta demande)
dt = 0.25 * dx**2  #Pas de temps
r = dt/(2*dx**2) #coefficient r trouvé lors de l'établissement de l'algorithme
tmax = 5000 #itérations maximum de simulation

"""Conditions initiales"""
x = np.linspace(-L, L, Nx) 
V = np.zeros(Nx)
psi0 = 1.0
xi = 1.0 / np.sqrt(g)

# --- MODIFICATION ICI : EMPÊCHER LES PHONONS ---
# On crée une enveloppe qui lisse les bords pour éviter le choc numérique
c_s = np.sqrt(g * psi0**2)  # Vitesse du son
v_s = 0.4 * c_s             # Vitesse du soliton (ex: 40% de la vitesse du son)
v_rel = v_s / c_s
gamma = np.sqrt(1 - v_rel**2)
enveloppe = np.exp(-(x / (0.85 * L))**20) 
v = 100
# On applique l'enveloppe au profil tanh
psi = 1j * v + gamma * (np.tanh(gamma * x / (np.sqrt(2) * xi))).astype(complex) * enveloppe
# -----------------------------------------------

psi_prev = psi.copy()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5)) 
line, = ax1.plot(x, np.abs(psi)**2, lw=2) 
ax1.set_ylim(0, 1.2) # Modifié légèrement pour voir le sommet du condensat
ax1.set_ylabel(r'$|\Psi|²$')
ax1.set_xlabel('Position(m)')
ax1.set_title(f"Simulation GPE 1D (g = {g})")

norms = [] 
times = [] 
line2, = ax2.plot([], [], lw=2, color='red') 
ax2.set_ylim(0, 150) # Ajusté selon ta longueur L
ax2.set_title(r"Norme de $\Psi$") 
ax2.set_xlabel("Temps(s)") 

energies = [] 
line3, = ax3.plot([],[], lw=2, color ='green')
ax3.set_title(r"Energie de $\Psi$") 
ax3.set_xlabel("Temps(s)") 

# Construction des matrices A et B
Principale_DiagA = (1 + 1j*r + 0.5j*dt*V) * np.ones(Nx)
Diags_Inf_SuppA = -0.5j*r * np.ones(Nx-1)
A = diags([Diags_Inf_SuppA, Principale_DiagA, Diags_Inf_SuppA], [1,0,-1]).tocsc()

Principale_DiagB = (1 - 1j*r - 0.5j*dt*V) * np.ones(Nx)
Diags_Inf_SuppB = 0.5j*r * np.ones(Nx-1)
B = diags([Diags_Inf_SuppB, Principale_DiagB, Diags_Inf_SuppB], [1,0,-1]).tocsc()

def calculate_norm(psi):
    return np.sum(np.abs(psi)**2 * dx)

def calculate_energy(psi, psi_prev):
    psi_extrap = 1.5 * psi - 0.5 * psi_prev
    extrap_density = np.abs(psi_extrap)**2 
    grad2_psi = np.gradient(np.gradient(psi_extrap, dx), dx)
    grad_density = -np.real(np.conj(psi_extrap) * grad2_psi) 
    current_energy = np.sum((1/2 * grad_density + V * extrap_density + g/2 * extrap_density**2)* dx) 
    return current_energy

steps_per_frame = 100
phys_time = 0

def animate(i):
    global psi, psi_prev, phys_time

    for _ in range(0, steps_per_frame):
        extrap_density = np.abs((1.5 * psi - 0.5 * psi_prev))**2 
        Principale_DiagO = 0.5j*dt*g*extrap_density * np.ones(Nx)
        O = diags([Principale_DiagO],[0]).tocsc() 

        Aprime = A + O 
        Bprime = B - O 
        res = Bprime.dot(psi) 
        
        psi_next = spsolve(Aprime, res) 

        psi_prev = psi.copy() 
        psi = psi_next.copy()
        phys_time += dt

    norms.append(calculate_norm(psi))
    energies.append(calculate_energy(psi, psi_prev))
    times.append(phys_time)

    if len(times) > 1:
        ax2.set_xlim(0, times[-1])
        ax3.set_xlim(0, times[-1])
        ax2.set_ylim(min(norms) * 0.99, max(norms) * 1.01) 
        ax3.set_ylim(min(energies) * 0.99, max(energies) * 1.01)
    
    line.set_ydata(np.abs(psi)**2) 
    line2.set_data(times, norms) 
    line3.set_data(times, energies)
    return line, line2, line3

ani = FuncAnimation(fig, animate, frames=500, interval=20, blit=False)
plt.tight_layout()
plt.show()