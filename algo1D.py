import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from matplotlib.animation import FuncAnimation

L =  20
Nx = 500
dx = (2*L)/Nx
dt = 0.1


g = 100
r = dt/(2*dx**2)

x = np.linspace(-L, L, Nx)

psi = np.exp((-x**2)/2).astype(complex)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)

psi_prev = psi.copy()

V = (x**2)/2

fig, ax = plt.subplots()
line, = ax.plot(x, np.abs(psi)**2, lw=2)
ax.set_ylim(0, 1)
ax.set_ylabel('|psi|**2')
ax.set_xlabel('x')
#title = ax.set_title(f"Simulation GPE 1D (g = {g})")
txt_g = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontweight='bold')

def animate(i):

    global psi
    global psi_prev




    extrap_density = np.abs((1.5 * psi - 0.5 * psi_prev))**2
    #extrap_density /= np.sqrt(np.sum(extrap_density * dx))

    psi_prev = psi.copy()

    Principale_DiagA = (1 + 1j*r + 0.5j*dt*(V + g*extrap_density)) * np.ones(Nx)
    Diags_Inf_SuppA = -0.5j*r * np.ones(Nx-1)

    A = diags([Diags_Inf_SuppA, Principale_DiagA, Diags_Inf_SuppA], [1,0,-1]).toarray()

    Principale_DiagB = (1 - 1j*r - 0.5j*dt*(V + g*extrap_density)) * np.ones(Nx)
    Diags_Inf_SuppB = 0.5j*r * np.ones(Nx-1)


    B = diags([Diags_Inf_SuppB, Principale_DiagB, Diags_Inf_SuppB], [1,0,-1]).toarray()

    

    psi_next = spsolve(A, B @ psi)

    psi_prev = psi.copy()
    psi = psi_next.copy()

    

    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)
    line.set_ydata(np.abs(psi)**2)
    txt_g.set_text(f"Simulation GPE 1D (g = {g})")
    return line, txt_g

ani = FuncAnimation(fig, animate, frames=200, interval=20, blit=True)
plt.show()
