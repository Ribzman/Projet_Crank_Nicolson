import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from matplotlib.animation import FuncAnimation

tmax = 5000
Nx = 64
Ny = Nx
N = Nx * Ny
L=10
g = 100
x = np.linspace(-L, L, Nx)
y = np.linspace(-L, L, Ny)
dx = np.sqrt(1/g) * (2*L/Nx)
dy = np.sqrt(1/g) * (2*L/Ny)

dt= 0.5 * dx**2
sigma = 4
rx = -dt/(4j*dx**2)
ry = -dt/(4j*dy**2)

X, Y = np.meshgrid(x, y)

theta1 =np.arctan2(Y+3, X-3)
theta2 =np.arctan2(Y-3, X+3)
theta3 = np.arctan2(Y,X)

psi = np.exp(0.5* (-(X**2+Y**2))/(2*sigma**2))*np.exp(1j*theta3).astype(complex)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)
psi2D = psi.flatten().astype(complex)

psi2D_prev = psi2D.copy()


V = 0.5 *  (X**2 + Y**2)
V0 = (V).flatten()

mask = np.ones(N-1)
mask[Nx-1::Nx] = 0

Principale_DiagA = (1 + 2*rx + 2*ry + 0.5j*dt*(V0)) * np.ones(N)
Diags_Inf_SuppA = -rx * mask
Diags_DistantesA = -ry * np.ones(N-Ny)

A = diags([Diags_DistantesA, Diags_Inf_SuppA, Principale_DiagA, Diags_Inf_SuppA, Diags_DistantesA], [Nx, 1,0,-1, -Nx], format='csc')
        
Principale_DiagB = (1 - 2*rx - 2*ry - 0.5j*dt*(V0)) * np.ones(N)
Diags_Inf_SuppB = rx * mask
Diags_DistantesB = ry * np.ones(N-Ny)

B = diags([Diags_DistantesB, Diags_Inf_SuppB, Principale_DiagB, Diags_Inf_SuppB, Diags_DistantesB], [Nx, 1,0,-1, -Nx], format='csc')


fig, axes = plt.subplots(2, 3, figsize=(10, 10))
ax1, ax2, ax_empty = axes[0]
ax3, ax4, ax5      = axes[1]

ax_empty.set_visible(False)  
extent = [-L, L, -L, L]

im  = ax1.imshow(np.abs(psi2D.reshape(Nx, Ny))**2,
                  interpolation='bilinear', extent=extent,
                  origin='lower', cmap='magma')
ang = ax2.imshow(np.angle(psi2D.reshape(Nx, Ny)),
                  norm=plt.Normalize(-np.pi, np.pi),
                  interpolation='nearest', extent=extent,
                  origin='lower', cmap='hsv')
plt.colorbar(im,  ax=ax1, label='Densité |ψ|²')
plt.colorbar(ang, ax=ax2, label='Phase')
ax1.set_title('Densité |ψ|²')
ax2.set_title('Phase ψ')

norms = []
times = []
energies = []

Norm, = ax3.plot([], [])
ax3.set_title('norme de Psi')

Energy,  = ax4.plot([], [])
ax4.set_title('Partie réelle Re(ψ)')

im_im,  = ax5.plot([],[])
ax5.set_title('Partie imaginaire Im(ψ)')

fig.suptitle("Vortex precessions in 2D")

step = 1
phys_time = 0

def calculate_norm(psi2D):
    N= np.sum(np.abs(psi2D.reshape(Nx,Ny))**2 * dx * dy)
    return N

def calculate_energy(psi2D, psiExtrap):
    p = psi2D.reshape(Nx, Ny)
    grad_y, grad_x = np.gradient(p, dy, dx)
    kinetic = 0.5 * np.sum(np.abs(grad_x)**2 + np.abs(grad_y)**2) * dx * dy
    potential = np.sum(V * np.abs(p)**2) * dx * dy
    interaction = 0.5 * g * np.sum(np.abs(psiExtrap)**4) * dx * dy
    return np.real(kinetic + potential + interaction)

def animate(i):

    global psi2D
    global psi2D_prev
    global phys_time

    for j in range(step):

        psiExtrap = 1.5 * psi2D - 0.5 * psi2D_prev
        extrap_density = np.abs(psiExtrap)**2

        Principale_DiagO = 0.5j * dt * g * extrap_density * np.ones(N)
        O = diags([Principale_DiagO], [0], format='csc')

        psi2D_next = spsolve((A+O), (B-O) @ psi2D)

        psi2D_prev = psi2D.copy()
        psi2D = psi2D_next.copy()

        density_2d = np.abs(psi2D.reshape(Nx, Ny))**2
        phase = np.angle(psi2D.reshape(Nx, Ny))

        phys_time += dt


    norms.append(calculate_norm(psi2D))
    energies.append(calculate_energy(psi2D, psiExtrap))
    times.append(phys_time*g)

    ax3.relim()       
    ax3.autoscale_view()
    ax4.relim()
    ax4.autoscale_view()

    im.set_array(density_2d)
    im.set_clim(0, np.max(density_2d))
    ang.set_array(phase)
    ang.set_clim(-np.pi, np.pi)
    Norm.set_data(times, norms)
    Energy.set_data(times, energies)

    return[im, ang, Norm,Energy]
    
print(np.max(np.abs(psi2D)**2))

psi_final = np.abs(psi2D.reshape(Nx, Ny)) ** 2

print(np.max((psi_final)))

ani = FuncAnimation(fig, animate, frames=tmax, interval=1, blit=False)
plt.tight_layout()
plt.show()