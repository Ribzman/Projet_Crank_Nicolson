import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from matplotlib.animation import FuncAnimation

tmax = 5000
Nx = 64
Ny = Nx
N  = Nx * Ny
L  = 30
g  = 100

x  = np.linspace(-L, L, Nx)
y  = np.linspace(-L, L, Ny)
dx = np.sqrt(1/g) * (2*L/Nx)
dy = np.sqrt(1/g) * (2*L/Ny)
dt = 0.25 * dx**2

rx = -dt / (4j * dx**2)
ry = -dt / (4j * dy**2)

X, Y = np.meshgrid(x, y)

sigma = 4
theta = np.arctan2(Y-3, X+3)
theta2 = np.arctan2(Y+3, X-3)

psi = np.exp(-0.5 * (X**2 + Y**2) / (2*sigma**2)) * np.exp(1j*theta).astype(complex)
psi2D = psi.flatten()

V  = 0.5 * (X**2 + Y**2)
V0 = V.flatten()

Principale_DiagA = (1 + 2*rx + 2*ry + 0.5j*dt*V0) * np.ones(N)
offA  = -rx * np.ones(N-1)
farA  = -ry * np.ones(N-Ny)
A = diags([farA, offA, Principale_DiagA, offA, farA],[Nx, 1, 0, -1, -Nx], format='csc')

Principale_DiagB = (1 - 2*rx - 2*ry - 0.5j*dt*V0) * np.ones(N)
offB  =  rx * np.ones(N-1)
farB  =  ry * np.ones(N-Ny)
B = diags([farB, offB, Principale_DiagB, offB, farB], [Nx, 1, 0, -1, -Nx], format='csc')


def calculate_norm(psi2D):
    return np.sum(np.abs(psi2D.reshape(Nx, Ny))**2) * dx * dy


def calculate_energy(psi2D, psi_prev):
    p_extrap = (1.5 * psi2D - 0.5 * psi_prev).reshape(Nx,Ny)
    density = np.abs(p_extrap)**2

    d2x = np.gradient(np.gradient(p_extrap, dx, axis=1), dx, axis=1)
    d2y = np.gradient(np.gradient(p_extrap, dy, axis=0), dy, axis=0)
    laplacian = d2x + d2y

    E1 = 0.5 * np.sum(-np.real(np.conj(p_extrap) * laplacian)) * dx * dy
    E2 = np.sum(V * density) * dx * dy
    E3 = 0.5*g * np.sum(density**2) * dx * dy

    return np.real(E1+E2+E3)

def calculate_angular_momentum(psi2D):
    p = psi2D.reshape(Nx,Ny)

    gradx = np.gradient(p, dx, axis=1)
    grady = np.gradient(p, dy, axis=0)

    i = np.conj(p) * (-1j) * (X * grady - Y * gradx)

    return np.real(np.sum(i* dx* dy))

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
ax1, ax2, ax_empty = axes[0]
ax3, ax4, ax5      = axes[1]
ax_empty.set_visible(False)
ax5.set_visible(True)
extent = [-L, L, -L, L]

im  = ax1.imshow(np.abs(psi2D.reshape(Nx, Ny))**2, interpolation='bilinear', extent=extent, origin='lower', cmap='magma')
ang = ax2.imshow(np.angle(psi2D.reshape(Nx, Ny)), norm=plt.Normalize(-np.pi, np.pi), interpolation='nearest', extent=extent, origin='lower', cmap='hsv')

plt.colorbar(im,  ax=ax1, label='Density |ψ|²')
plt.colorbar(ang, ax=ax2, label='Phase')
ax1.set_title('Density |ψ|²')
ax2.set_title('Phase ψ')

norms    = []
times    = []
energies = []
angular_momentums = []

Norm,   = ax3.plot([], [], lw=1.2)
Energy, = ax4.plot([], [], lw=1.2, color='Green')
AngMom, = ax5.plot([],[], lw= 1.2, color='Red')
ax3.set_title('Norm ‖ψ‖²')
ax4.set_title('Energy E(t)')
ax5.set_title('Angular Momentum')

phys_time = 0
step = 1


def animate(i):
    global psi2D, phys_time

    for _ in range(step):
        psi_prev = psi2D.copy()
        avg_density = np.abs(1.5 * psi2D - 0.5 * psi_prev)**2
        O = diags([0.5j * dt * g * avg_density], [0], format='csc')
        psi_next = spsolve((A + O), (B - O) @ psi2D)

        psi2D = psi_next
        phys_time += dt

    

    density_2d = np.abs(psi2D.reshape(Nx, Ny))**2
    phase      = np.angle(psi2D.reshape(Nx, Ny))

    
    norms.append(calculate_norm(psi2D))
    energies.append(calculate_energy(psi2D, psi_prev))
    angular_momentums.append(calculate_angular_momentum(psi2D))
    times.append(phys_time * g)
    
    

    im.set_array(density_2d)
    im.set_clim(0, np.max(density_2d))
    ang.set_array(phase)
    ang.set_clim(-np.pi, np.pi)

    Norm.set_data(times, norms)
    Energy.set_data(times, energies)
    AngMom.set_data(times, angular_momentums)

    ax3.set_ylim(0, 2 * np.max(norms))
    ax4.set_ylim(0, 2 * np.max(energies))
    ax5.set_ylim(0, 2 * np.max(angular_momentums))
    ax3.set_xlim(0, 2 * np.max(times))
    ax4.set_xlim(0, 2 * np.max(times))
    ax5.set_xlim(0, 2 * np.max(times))


    return [im, ang, Norm, Energy, AngMom]


ani = FuncAnimation(fig, animate, frames=tmax, interval=1, blit=False)
plt.tight_layout()
plt.show()