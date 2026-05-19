import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from matplotlib.animation import FuncAnimation

"""Constantes Physiques"""
tmax = 5000 #nombre maximal d'itération de la boucle d'animation
Nx, Ny = 100, 100 #longueur et largeur de la grille 
N  = Nx * Ny #taille totale de la grille aplatie en 1D
L  = 30 #demi-taille physique du système (domaine de -L à +L)
g  = 10 #constante d'interaction entre les bosons 
phys_time = 0 #temps physique cumulé
step = 1 #nombre de pas de temps effectués par frame

"""Discrètisation de l'éspace et du temps"""
x  = np.linspace(-L, L, Nx) #axe spatial discrétisé selon x
y  = np.linspace(-L, L, Ny) #axe spatial discrétisé selon y
dx = np.sqrt(1/g) * (2*L/Nx) #pas spatial en x
dy = np.sqrt(1/g) * (2*L/Ny) #pas spatial en y
dt = 0.1 * dx**2 #pas de temps

rx = -dt / (4j * dx**2) #coefficient de Crank-Nicolson selon x
ry = -dt / (4j * dy**2) #coefficient de Crank-Nicolson selon y

X, Y = np.meshgrid(x, y) #grille 2D de coordonnées spatiales

"""Condition Initiale"""
sigma  = 4 #largeur du paquet d'onde gaussien initial
theta  = np.arctan2(Y, X) #angle centré à l'origine (charge +1)
theta2 = np.arctan2(Y+2, X-2) #angle centré en un point (2,-2) (charge +1)
theta2 = np.arctan2(Y-2, X+2) #angle centré en un point (-2,2) (charge +1)
psi    = np.exp(-0.5 * (X**2 + Y**2) / (2*sigma**2)) * np.exp(1j*theta).astype(complex) #gaussienne modulée par une phase de vortex
psi2D  = psi.flatten() #mise à plat en vecteur 1D pour la résolution matricielle

"""Potentiel"""
V  = 0.5 * (X**2 + Y**2) #potentiel de piègeage harmonique
V0 = V.flatten() #version aplatie pour la construction des matrices

"""Creations des matrice A et B"""
Principale_DiagA = (1 + 2*rx + 2*ry + 0.5j*dt*V0) * np.ones(N) #diagonale principale de A
Diags_Inf_SuppA  = -rx * np.ones(N-1) #Diagonales inferieur et superieur de B
DiagDistanteA  = -ry * np.ones(N-Ny)  #couplage avec les voisins distants selon y
A = diags([DiagDistanteA, Diags_Inf_SuppA, Principale_DiagA, Diags_Inf_SuppA, DiagDistanteA],[Nx, 1, 0, -1, -Nx], format='csc') #création de la matrice creuse A
Principale_DiagB = (1 - 2*rx - 2*ry - 0.5j*dt*V0) * np.ones(N) #diagonale principale de B
Diags_Inf_SuppB  =  rx * np.ones(N-1) #Diagonales inferieur et superieur de B
DiagDistanteB  =  ry * np.ones(N-Ny) #Diagonales distantes de B
B = diags([DiagDistanteB, Diags_Inf_SuppB, Principale_DiagB, Diags_Inf_SuppB, DiagDistanteB], [Nx, 1, 0, -1, -Nx], format='csc') #création de la matrice creuse B


"""Fonction calculant la norme de la fonction d'onde psi"""
def calculate_norm(psi2D):
    return np.sum(np.abs(psi2D.reshape(Nx, Ny))**2) * dx * dy

"""Fonction calculant l'énergie de la fonction psi"""
def calculate_energy(psi2D, psi_prev):
    p_extrap = (1.5 * psi2D - 0.5 * psi_prev).reshape(Nx,Ny) 
    density = np.abs(p_extrap)**2

    d2x = np.gradient(np.gradient(p_extrap, dx, axis=1), dx, axis=1)
    d2y = np.gradient(np.gradient(p_extrap, dy, axis=0), dy, axis=0)
    laplacian = d2x + d2y

    Ec = 0.5 * np.sum(-np.real(np.conj(p_extrap) * laplacian)) * dx * dy #énergie cinétique
    Ep = np.sum(V * density) * dx * dy #énergie potentielle
    Ei = 0.5*g * np.sum(density**2) * dx * dy #énergie d'interaction

    return np.real(Ec+Ep+Ei)  #énergie totale du condensat

"""Fonction Calculant le moment angulaire de la fonction d'onde psi"""
def calculate_angular_momentum(psi2D):
    p = psi2D.reshape(Nx,Ny) #reshape en 2D pour le calcul des gradients

    gradx = np.gradient(p, dx, axis=1) #gradient de psi selon x
    grady = np.gradient(p, dy, axis=0) #gradient de psi selon y

    i = np.conj(p) * (-1j) * (X * grady - Y * gradx) #intégrande

    return np.real(np.sum(i* dx* dy))

"""Fonction qui construit la matrice O"""
def construct_O(psi2D, psi_prev):
    avg_density = np.abs(1.5 * psi2D - 0.5 * psi_prev)**2 #densité extrapolée à mi-pas pour le terme non-linéaire
    return diags([0.5j * dt * g * avg_density], [0], format='csc') #diagonale du terme d'interaction semi-implicite

"""Initialisation de la figure, des graphiques et des axes"""
fig, axes = plt.subplots(2, 3, figsize=(12, 8)) #grille de 6 sous-graphes
ax1, ax2, ax_empty = axes[0] #ligne du haut : densité, phase, case vide
ax3, ax4, ax5      = axes[1] #ligne du bas  : norme, énergie, moment cinétique
ax_empty.set_visible(False) #masquage de la case vide
extent = [-L, L, -L, L] #étendue spatiale commune aux images 2D

im  = ax1.imshow(np.abs(psi2D.reshape(Nx, Ny))**2, interpolation='bilinear', extent=extent, origin='lower', cmap='magma') #affichage de la densité de psi
ang = ax2.imshow(np.angle(psi2D.reshape(Nx, Ny)), norm=plt.Normalize(-np.pi, np.pi), interpolation='nearest', extent=extent, origin='lower', cmap='hsv')  #affichage de la phase de psi

plt.colorbar(im,  ax=ax1, label='Density |ψ|²') #barre de couleur pour la densité
plt.colorbar(ang, ax=ax2, label='Phase') #barre de couleur pour la phase
ax1.set_title(r'Density $|~\psi~|^2$') #titre du graphe de densité
ax2.set_title(r'Phase $\psi$') #titre du graphe de phase

norms = [] #historique de la norme
times = [] #historique des temps
energies = [] #historique de l'énergie
angular_momentums = [] #historique du moment cinétique

Norm,   = ax3.plot([], [], lw=1.2) #courbe de la norme
Energy, = ax4.plot([], [], lw=1.2, color='Green') #courbe de l'énergie
AngMom, = ax5.plot([],[], lw= 1.2, color='Red')  #courbe du moment cinétique
ax3.set_title(r'Norm $|| \psi ||^2$')         #titre du graphe de norme
ax4.set_title(r'Energy $E(t)$')        #titre du graphe d'énergie
ax5.set_title(r'Angular Momentum $L_z$')   #titre du graphe de moment cinétique

"""Boucle d'animation"""
def animate(i):
    global psi2D, phys_time  #variables globales modifiées à chaque frame

    for _ in range(step):
        psi_prev = psi2D.copy() #sauvegarde de l'intération précédente
        
        O = construct_O(psi2D, psi_prev)
        psi_next = spsolve((A + O), (B - O) @ psi2D) #résolution du système linéaire creux

        psi2D = psi_next #mise à jour de la fonction d'onde
        phys_time += dt #avancement du temps physique

    density_2d = np.abs(psi2D.reshape(Nx, Ny))**2 #densité 2D à afficher
    phase = np.angle(psi2D.reshape(Nx, Ny)) #phase 2D à afficher

    norms.append(calculate_norm(psi2D)) #enregistrement de la norme
    energies.append(calculate_energy(psi2D, psi_prev)) #enregistrement de l'énergie
    angular_momentums.append(calculate_angular_momentum(psi2D)) #enregistrement du moment cinétique
    times.append(phys_time) #temps physique

    im.set_array(density_2d) #mise à jour de l'image de densité
    im.set_clim(0, np.max(density_2d)) #renormalisation dynamique de la colormap
    ang.set_array(phase) #mise à jour de l'image de phase
    ang.set_clim(-np.pi, np.pi) #limites fixes de la colormap de phase

    Norm.set_data(times, norms) #mise à jour de la courbe de norme
    Energy.set_data(times, energies) #mise à jour de la courbe d'énergie
    AngMom.set_data(times, angular_momentums) #mise à jour de la courbe de moment cinétique

    """Réglage des axes"""
    ax3.set_ylim(0, 2 * np.max(norms))
    ax4.set_ylim(0, 2 * np.max(energies))           
    ax5.set_ylim(0, 2 * np.max(angular_momentums))  
    ax3.set_xlim(0, 2 * np.max(times))              
    ax4.set_xlim(0, 2 * np.max(times))              
    ax5.set_xlim(0, 2 * np.max(times))              

    return [im, ang, Norm, Energy, AngMom]  #retour des variables mises à jour


ani = FuncAnimation(fig, animate, frames=tmax, interval=1, blit=False) #lancement de l'animation
plt.tight_layout() #ajustement automatique de la mise en page
plt.show() #affichage de la fenêtre