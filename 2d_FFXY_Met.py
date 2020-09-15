import numpy as np
import matplotlib.pyplot as pyplot
from mpl_toolkits import mplot3d
import os
import sys
import time

# 2d FFXY (rigid rotor) spin model
# Spins sit at lattice sites of square lattice
# Metropolis algorithm

L = 20  # Size of grid is LxL (L^2 total spins)

sigPhi = np.pi / 4  # stDev for distribution of angle changes

# T = 0.01  # Temperature
T = float(sys.argv[1]) / float(sys.argv[2])   # Read in temp from cp args

K = 40  # Average number of flips per spin

display = False    # Display final spin configuration
save = True    # Save final spin configuration to file
progress = True     # Print progress during flips


# Change to energy from changing spin at (x,y) by angle phi
def deltaE(g, x, y, phi):
    nbrs = [[0, 1], [0, -1], [1, 0], [-1, 0]]

    dE = 0
    for i in range(4):
        nbr = (np.array([x, y]) + nbrs[i]) % L
        fij = (y % 2) * nbrs[i][0] * np.pi
        dE += np.cos(g[x, y] - g[nbr[0], nbr[1]] - fij)
        dE -= np.cos(g[x, y] - g[nbr[0], nbr[1]] + phi - fij)

    return dE


# Total energy of configuration
def energy(g):
    nrg = 0
    for x in range(L):
        for y in range(L):
            nrg -= np.cos(g[x, y] - g[(x + 1) % L, y])
            nrg -= np.cos(g[x, y] - g[x, (y + 1) % L])
    return nrg / (L ** 2)


# Susceptibility of configuration
def suscept(g):
    mx = np.average(np.cos(g))
    my = np.average(np.sin(g))
    return mx * mx + my * my


# Initialize grid of spins
grid = 2 * np.pi * np.random.randn(L, L)

# Keep track of susceptibility and energy as a diagnostic
# for reaching thermal equilibrium
chi = [suscept(grid)]
e = [energy(grid)]


flips = 0

while flips < K * L * L:
    if progress & (flips % 100 == 0):
        print("%4.1f" % (100 * flips / (K * L * L)), "%")

    # Random spin and random proposed change to angle
    x, y = np.random.randint(0, L, 2)
    phi = sigPhi * np.random.randn()

    # Accept change with probability exp(-ΔE/T)
    if (np.random.rand() < np.exp(-deltaE(grid, x, y, phi) / T)):
        grid[x, y] += phi
        flips += 1

    # Record susceptibility and energy
    chi = np.append(chi, suscept(grid))
    e = np.append(e, energy(grid))


if display:
    # fig, ax = pyplot.subplots(1, 2, figsize=(10, 4))
    # ax[0].plot(chi)
    # ax[1].plot(e)
    # pyplot.show()

    vx = np.cos(grid)
    vy = np.sin(grid)
    pyplot.quiver(vx, vy, grid)
    pyplot.show()

    # x = y = range(L)
    # x, y = np.meshgrid(x, y)
    # z = np.mod(grid + np.pi, 2 * np.pi) - np.pi

    # fig = pyplot.figure()
    # ax = pyplot.axes(projection='3d')
    # # ax.set_zlim(0, 2 * np.pi)
    # ax.set_zlim(-np.pi, np.pi)
    # ax.scatter(x, y, z, c=z.flatten())
    # pyplot.show()


if save:
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    path = os.path.join(fileDir, 'Data_FFXY')
    pathLK = os.path.join(fileDir, 'Data_FFXY/L=%s_K=%s' % (L, K))
    pathLKT = os.path.join(fileDir, 'Data_FFXY/L=%s_K=%s/T=%.2f' % (L, K, T))
    filename = os.path.join(fileDir, 'Data_FFXY/L=%s_K=%s/T=%.2f/%s.txt'
                            % (L, K, T, int(time.time())))

    # Create folders if necessary
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(pathLK):
        os.makedirs(pathLK)
    if not os.path.exists(pathLKT):
        os.makedirs(pathLKT)

    np.savetxt(filename, np.mod(grid, 2 * np.pi), fmt='%.4f')
