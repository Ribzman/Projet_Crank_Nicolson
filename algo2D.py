import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from matplotlib.animation import FuncAnimation


Nx = 64
Ny = 64
N = Nx * Ny
L=10
g = 1
x = np.linspace(-L, L, Nx)
y = np.linspace(-L, L, Ny)
dx = np.sqrt(1/g) * (2*L/Nx)
dy = np.sqrt(1/g) * (2*L/Ny)

V0=200

dt= 0.25 * dx**2
sigma = 4
rx = -dt/(4j*dx**2)
ry = -dt/(4j*dy**2)

X, Y = np.meshgrid(x, y)

theta1 =np.arctan2(Y+3, X-3)
theta2 =np.arctan2(Y-3, X+3)
theta3 = np.arctan2(Y-2,X)

psi = np.exp(0.5* (-(X**2+Y**2))/(2*sigma**2))*np.exp(1j*theta3).astype(complex)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)
psi2D = psi.flatten().astype(complex)

psi2D_prev = psi2D.copy()


V = 0.5 *  (X**2 + Y**2)
V0 = (V).flatten()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
norm = plt.Normalize(0,1)
im = ax1.imshow(np.abs(psi2D.reshape(Nx, Ny))**2, norm = norm, interpolation='bilinear', extent=[-L, L, -L, L], origin='lower', cmap='magma')
ang = ax2.imshow(np.angle(psi2D.reshape(Nx, Ny)), interpolation='nearest',extent=[-L, L, -L, L], origin='lower', cmap='hsv')
plt.colorbar(im, label='Density |ψ|²')
plt.colorbar(ang, label='Phase')
fig.suptitle("Vortex precessions in 2D")

step = 1

def animate(i):

    global psi2D
    global psi2D_prev

    for i in range(step):

        extrap_density = np.abs(1.5 * psi2D - 0.5 * psi2D_prev)**2

        psi2D_prev = psi2D.copy()

        Principale_DiagA = (1 + 2*rx + 2*ry + 0.5j*dt*(V0 + g*extrap_density)) * np.ones(N)
        Diags_Inf_SuppA = -rx * np.ones(N-1)
        Diags_DistantesA = -ry * np.ones(N-Ny)

        A = diags([Diags_DistantesA, Diags_Inf_SuppA, Principale_DiagA, Diags_Inf_SuppA, Diags_DistantesA], [Nx, 1,0,-1, -Nx], format='csc')
        
        Principale_DiagB = (1 - 2*rx - 2*ry + 0.5j*dt*(V0 + g * extrap_density)) * np.ones(N)
        Diags_Inf_SuppB = rx * np.ones(N-1)
        Diags_DistantesB = ry * np.ones(N-Ny)

        B = diags([Diags_DistantesB, Diags_Inf_SuppB, Principale_DiagB, Diags_Inf_SuppB, Diags_DistantesB], [Nx, 1,0,-1, -Nx], format='csc')

        

        psi2D_next = spsolve(A, B @ psi2D)

        psi2D_prev = psi2D.copy()
        psi2D = psi2D_next.copy()

        
        
        psi2D /= np.sqrt(np.sum(np.abs(psi2D)**2) * dx* dy)
        #psi2D /= np.max(np.abs(psi2D)**2)

        density_2d = np.abs(psi2D.reshape(Nx, Ny))**2
        phase = np.angle(psi2D.reshape(Nx, Ny))
        
    im.set_array(density_2d)
    im.set_clim(0, np.max(density_2d))
    ang.set_array(phase)
    ang.set_clim(-np.pi, np.pi)

    return[im, ang]
    
print(np.max(np.abs(psi2D)**2))
psi2D /= np.sqrt(np.sum(np.abs(psi2D)**2) * dx* dy)

psi_final = np.abs(psi2D.reshape(Nx, Ny)) ** 2
#psi_final /= np.max(np.abs(psi_final)**2)

print(np.max((psi_final)))

ani = FuncAnimation(fig, animate, frames=200, interval=1, blit=True)
plt.tight_layout()
plt.show()