import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.perf_counter()

L = 14          # grid parameters
N = 128

kke = 1.0             # kinetic coefficient
kne = -1.0            # electron-nuclear attraction coefficient
kee = 1.0             # electron-electron repulsion coefficient
Z = -kne              # nuclear charge
bond = 1
dtau = 0.05           # iteration parameters
ImIter = 300
HFIter = 600

# bond samples stay on the CPU - just a small 1D scan array, no need for GPU
bondSmples = np.append(np.linspace(0, 1, 20), np.linspace(1, 5, 10))

sigma = 2.0           # wavefunction initialization

# ----- build the grid on the GPU directly -----
x = y = cp.linspace(-L, L, N)       # initialize spatial grid
X, Y = cp.meshgrid(x, y)
dx = float(x[1] - x[0])
dy = float(y[1] - y[0])

kx = 2 * cp.pi * cp.fft.fftfreq(N, d=dx)   # initialize frequency grid
ky = 2 * cp.pi * cp.fft.fftfreq(N, d=dy)
KX, KY = cp.meshgrid(kx, ky, indexing="ij")

expT = cp.exp(-0.5 * dtau * (KX**2 + KY**2))   # kinetic propagator

eps = 0.02
G = 1 / cp.sqrt(X**2 + Y**2 + eps)   # Green's function

psi = cp.exp(-(X - bond)**2 / sigma - Y**2 / sigma)   # Initialize Psi
psi /= cp.sqrt(cp.sum(cp.abs(psi)**2) * dx * dy)      # Normalize Psi
psi = psi.astype(cp.complex128)

rhoA = cp.abs(psi)**2           # calculate density
rhoB = rhoA[:, ::-1]

# once, outside the loop
Npad = 2 * N  # linear-convolution padding size fftconvolve would use
G_hat = cp.fft.rfft2(G, s=(Npad, Npad))


def hartree(rho):
    rho_hat = cp.fft.rfft2(rho, s=(Npad, Npad))
    conv = cp.fft.irfft2(rho_hat * G_hat, s=(Npad, Npad))
    start = (Npad - N) // 2
    return conv[start:start + N, start:start + N] * dx * dy


Energies = []
bestEnergy = np.inf
bestDensity = None
bestBond = None

for bond in bondSmples:
    # iterate over bond samples
    Vne = kne * (1 / cp.sqrt((X - bond)**2 + Y**2 + eps)
                 + 1 / cp.sqrt((X + bond)**2 + Y**2 + eps))   # nuclear potential

    for _ in range(HFIter):
        rhoA_old = rhoA.copy()
        # iterate for self consistent field
        Vee_a = kee * hartree(rhoB)   # electron-electron potential
        expV = cp.exp(-0.5 * dtau * (Vee_a + Vne))   # potential operator

        for it in range(ImIter):
            psiOld = psi.copy()

            psi *= expV   # evolve imaginary time
            psi = cp.fft.ifft2(cp.fft.fft2(psi) * expT)
            psi *= expV
            psi /= cp.sqrt(cp.sum(cp.abs(psi)**2) * dx * dy)

            Im_err = float(cp.linalg.norm(cp.abs(psi)**2 - cp.abs(psiOld)**2))

            if Im_err < 1e-8:
                break

        rhoA = cp.abs(psi)**2
        rhoB = rhoA[:, ::-1]
        hf_err = float(cp.linalg.norm(rhoA - rhoA_old))

        if hf_err < 1e-8:
            print(f"HF Loop Converged after {it+1} iterations.")
            T = 2 * float(cp.real(cp.sum(cp.conj(psi) * cp.fft.ifft2(
                    0.5 * (KX**2 + KY**2) * cp.fft.fft2(psi))) * dx * dy))
            Ene = 2 * float(cp.sum(cp.abs(psi)**2 * Vne) * dx * dy)
            J = float(cp.sum(cp.abs(psi)**2 * Vee_a) * dx * dy)
            Enn = Z**2 / (2 * bond)

            HartreeEnergy = T + Ene + Enn + J
            Energies.append(HartreeEnergy)

            if HartreeEnergy < bestEnergy:
                bestEnergy = HartreeEnergy
                bestDensity = cp.asnumpy(rhoTotal)
                bestBond = bond

            break
    print(hf_err)

    end_time = time.perf_counter()
    rhoA = cp.abs(psi)**2
    rhoB = rhoA[:, ::-1]
    rhoTotal = rhoA + rhoB

    print(f"converged in {end_time-start_time}")
    start_time = time.perf_counter()


# Final densities (pulled back to host for plotting)


rhoA = cp.abs(psi)**2
rhoB = rhoA[:, ::-1]
rhoTotal = cp.asnumpy(rhoA + rhoB)

# Plot
# Bonding curve

plt.figure(figsize=(7, 5))

plt.plot(
    2 * bondSmples,          # actual proton-proton separation
    Energies,
    "o-",
    lw=2
)

plt.xlabel("Bond Length (Bohr)")
plt.ylabel("Total Energy (Hartree)")
plt.title("UHF Bonding Curve")
plt.grid(True)

plt.tight_layout()
plt.show()

# Density at minimum-energy bond length

plt.figure(figsize=(7, 6))

plt.imshow(
    bestDensity,
    extent=[-L, L, -L, L],
    origin="lower",
    cmap="magma",
    interpolation="bicubic"
)

plt.scatter(
    [-bestBond, bestBond],
    [0, 0],
    marker="+",
    c="cyan",
    s=180,
    linewidths=2,
    label="Protons"
)

plt.xlabel("x (Bohr)")
plt.ylabel("y (Bohr)")
plt.title(f"Minimum-Energy Density\nBond Length = {2*bestBond:.3f} Bohr")
plt.colorbar(label=r"$\rho(\mathbf{r})$")
plt.legend()

plt.tight_layout()
plt.show()

print(f"Minimum Energy : {bestEnergy:.8f} Hartree")
print(f"Equilibrium Bond Length : {2*bestBond:.4f} Bohr")
