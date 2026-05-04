import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import os
from datetime import datetime

"""Constantes du problème"""
g = 400
L = 20
Nx = 2000
dx = np.sqrt(1/np.abs(g))*(2*L)/(Nx)
dt = 0.25 * dx**2
r = dt/(2*dx**2)
tmax = 5000
sigma = 1
steps_per_frame = 100
phys_time = 0

# --- Paramètres d'enregistrement ---
FPS = 10                          # Réduit pour alléger le GIF
DUREE_SEC = 80
TOTAL_FRAMES = FPS * DUREE_SEC    # = 600 frames
OUTPUT_DIR = "SimulationGPE1D"
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(OUTPUT_DIR, f"simulation_g{g}_{timestamp}.gif")

"""Conditions initiales"""
x = np.linspace(-L, L, Nx)
V = (x**2)/2
psi = np.exp((-x**2)/(2*sigma**2)).astype(complex)
psi_prev = psi.copy()

"""Graphiques"""
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 5))
line,  = ax1.plot(x, np.abs(psi)**2, lw=2)
ax1.set_ylim(0, 1)
ax1.set_ylabel(r'$|\Psi|²$')
ax1.set_xlabel('Position(m)')
ax1.set_title(f"Simulation GPE 1D (g = {g})")

norms, times, energies = [], [], []

line2, = ax2.plot([], [], lw=2, color='red')
ax2.set_ylim(0, 1)
ax2.set_title(r"Norme de $\Psi$")
ax2.set_xlabel("Temps(s)")
ax2.set_ylabel(r"Norme de $\Psi$")

line3, = ax3.plot([], [], lw=2, color='green')
ax3.set_title(r"Energie de $\Psi$")
ax3.set_xlabel("Temps(s)")
ax3.set_ylabel(r"Energie de $\Psi$")

"""Matrices A et B"""
Principale_DiagA = (1 + 1j*r + 0.5j*dt*V) * np.ones(Nx)
Diags_Inf_SuppA  = -0.5j*r * np.ones(Nx-1)
A = diags([Diags_Inf_SuppA, Principale_DiagA, Diags_Inf_SuppA], [1,0,-1]).tocsc()

Principale_DiagB = (1 - 1j*r - 0.5j*dt*V) * np.ones(Nx)
Diags_Inf_SuppB  = 0.5j*r * np.ones(Nx-1)
B = diags([Diags_Inf_SuppB, Principale_DiagB, Diags_Inf_SuppB], [1,0,-1]).tocsc()

def calculate_norm(psi):
    return np.sum(np.abs(psi)**2 * dx)

def calculate_energy(psi, psi_prev):
    psi_extrap     = 1.5 * psi - 0.5 * psi_prev
    extrap_density = np.abs(psi_extrap)**2
    grad2_psi      = np.gradient(np.gradient(psi_extrap, dx), dx)
    grad_density   = -np.real(np.conj(psi_extrap) * grad2_psi)
    return np.sum((0.5 * grad_density + V * extrap_density + g/2 * extrap_density**2) * dx)

def Construct_O(psi, psi_prev):
    extrap_density   = np.abs(1.5 * psi - 0.5 * psi_prev)**2
    Principale_DiagO = 0.5j*dt*g*extrap_density * np.ones(Nx)
    return diags([Principale_DiagO], [0]).tocsc()

def animate(i):
    global psi, psi_prev, phys_time

    for _ in range(steps_per_frame):
        O        = Construct_O(psi, psi_prev)
        Aprime   = A + O
        Bprime   = B - O
        res      = Bprime.dot(psi)
        psi_next = spsolve(Aprime, res)
        psi_prev = psi.copy()
        psi      = psi_next.copy()
        phys_time += dt

    norms.append(calculate_norm(psi))
    energies.append(calculate_energy(psi, psi_prev))
    times.append(phys_time)

    if len(times) > 1:
        ax2.set_xlim(0, times[-1])
        ax3.set_xlim(0, times[-1])
    if len(norms) > 1:
        ax2.set_ylim(min(norms) * 0.9, max(norms) * 1.1)
    if len(energies) > 1:
        ax3.set_ylim(min(energies) * 0.9, max(energies) * 1.1)

    line.set_ydata(np.abs(psi)**2)
    line2.set_data(times[-tmax:], norms[-tmax:])
    line3.set_data(times[-tmax:], energies[-tmax:])

    print(f"\rEnregistrement : frame {i+1}/{TOTAL_FRAMES} | t = {phys_time:.4f}", end="")

    return line, line2, line3

"""Enregistrement avec PillowWriter"""
ani = animation.FuncAnimation(fig, animate, frames=TOTAL_FRAMES, interval=1000//FPS, blit=False)

writer = PillowWriter(fps=FPS)
plt.tight_layout()

print(f"Enregistrement en cours → {output_path}")
ani.save(output_path, writer=writer)
print(f"\nGIF sauvegardé : {output_path}")
plt.close()