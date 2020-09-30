import numpy as np
#import matplotlib.pyplot as pyplot
#from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import time

# 3d Ising spin model
# Spins sit at lattice sites of square lattice.

N = 15  # Size of grid is NxNxN (N^3 total spins)

# T = 3.5   # Temperature
T = float(sys.argv[1]) / float(sys.argv[2])   # Read in temp from cp args

K = 20   # Iterate until K * (N^3) spins have been flipped in total

choices = [-1, 1]   # Values to choose from to initialize grid

display = False    # Display final spin configuration
save = True   # Save final spin configuration to file
progress = False   # Print progress during flips


# Returns randomized grid
def randomGrid():
    return np.random.choice(choices, [N, N, N])


# Returns a random lattice site
def randomSite():
    return np.random.randint(0, N, 3)


def update(g):
    x, y, z = randomSite()
    cluster = np.array([[x, y, z]])
    candidates = np.empty([0, 3], dtype=int)

    nbrs = (np.array([x, y, z]) + [[0, 0, 1], [0, 0, -1],
                                   [0, 1, 0], [0, -1, 0],
                                   [1, 0, 0], [-1, 0, 0]]) % N
    for nbr in nbrs:
        if (g[x, y, z] == g[nbr[0], nbr[1], nbr[2]]):
            candidates = np.append(candidates, [nbr], axis=0)

    while len(candidates) > 0:
        c = candidates[0]
        candidates = candidates[1:]
        if (cluster == c).all(1).any():
            continue
        elif (np.random.random() < 1 - np.exp(-2. / T)):
            cluster = np.append(cluster, [c], axis=0)
            nbrs = (c + [[0, 0, 1], [0, 0, -1],
                         [0, 1, 0], [0, -1, 0],
                         [1, 0, 0], [-1, 0, 0]]) % N
            for nbr in nbrs:
                if (g[c[0], c[1], c[2]] == g[nbr[0], nbr[1], nbr[2]]):
                    candidates = np.append(candidates, [nbr], axis=0)

    return cluster


def energy(g):
    nrg = 0
    for x in range(N):
        for y in range(N):
            for z in range(N):
                nrg -= g[x, y, z] * g[(x + 1) % N, y, z]
                nrg -= g[x, y, z] * g[x, (y + 1) % N, z]
                nrg -= g[x, y, z] * g[x, y, (z + 1) % N]
    return nrg / (N ** 3)

for counter in np.arange(1000):
    # Initialize grid of spins
    grid = randomGrid()

    # Keep track of average magnitization as a diagnostic
    # for reaching thermal equilibrium
    m = [np.average(grid)]
    e = [energy(grid)]

    flips = 0


    while flips < K * N ** 3:
        #print("%4.1f" % (100 * flips / (K * N ** 3)), "%")
        clust = update(grid)
        for c in clust:
            grid[c[0], c[1], c[2]] *= -1
        m = np.append(m, abs(np.average(grid)))
        e = np.append(e, abs(energy(grid)))
        flips += len(clust)


    if display or save:
        maj = np.average(grid)
        majSpins = np.empty([0, 3])
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    if grid[x, y, z] * maj > 0:
                        majSpins = np.append(majSpins, [[x, y, z]], axis=0)

    if save:
        # Scan through the final spin configuration and save the locations
        # of those spins which are (anti-)aligned with the majority

        fileDir = os.path.dirname(os.path.realpath('__file__'))
        pathN = os.path.join(fileDir, 'Data_3d_Ising_Wolff_N=%s_K=%s' % (N, K))
        pathNT = os.path.join(fileDir, 'Data_3d_Ising_Wolff_N=%s_K=%s/%s' % (N, K, T))
        filename = os.path.join(fileDir, 'Data_3d_Ising_Wolff_N=%s_K=%s/%s/%s.txt'
                                % (N, K, T, int(time.time())))

        # Create folders if necessary
        if not os.path.exists(pathN):
            os.makedirs(pathN)
        if not os.path.exists(pathNT):
            os.makedirs(pathNT)

        np.savetxt(filename, majSpins, fmt='%d')

'''if display:
    fig, axes = pyplot.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(abs(m))
    axes[1].plot(e)
    pyplot.show()

    fig = pyplot.figure()
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.scatter(majSpins[:, 0], majSpins[:, 1], majSpins[:, 2], marker='.', s=0.1, c='k')
    pyplot.show()'''
