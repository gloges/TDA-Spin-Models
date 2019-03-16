import numpy as np
import matplotlib.pyplot as pyplot
import os
import sys
import time

# 2d Ising spin model
# Spins sit at lattice sites of square lattice
# Wolff (cluster) algorithm

N = 50  # Size of grid is NxN (N^2 total spins)

# T = 1.5   # Temperature
T = float(sys.argv[1]) / float(sys.argv[2])   # Read in temp from cp args

K = 20   # Average number of flips per spin

choices = [-1, 1]   # Values to choose from to initialize grid

display = True    # Display final spin configuration
save = False    # Save final spin configuration to file
progress = True     # Print progress during flips


# Builds a cluster in a flood-fill-like way and returns list of spin locations.
# candidates is a list of neighboring spins which have yet to be tested/added
# to the cluster. A single spin may be in candidates more than once if it is
# adjacent to multiple cluster spins.
def update(g):
    # Seed spin
    x, y = np.random.randint(0, N, 2)
    cluster = np.array([[x, y]])

    # Start with those four spins which are next to the seed spin.
    candidates = np.empty([0, 2], dtype=int)
    nbrs = (np.array([x, y]) + [[0, 1], [0, -1], [1, 0], [-1, 0]]) % N
    for nbr in nbrs:
        if (g[x, y] == g[nbr[0], nbr[1]]):
            candidates = np.append(candidates, [nbr], axis=0)

    while len(candidates) > 0:
        # Take first candidate
        c = candidates[0]
        candidates = candidates[1:]

        if (cluster == c).all(1).any():
            # Check that this spin hasn't already been added to the cluster
            continue
        elif (np.random.random() < 1 - np.exp(-2. / T)):
            # Probabalistically add spin to cluster
            # and add its aligned neighbors to the list of candidates
            cluster = np.append(cluster, [c], axis=0)
            nbrs = (c + [[0, 1], [0, -1], [1, 0], [-1, 0]]) % N
            for nbr in nbrs:
                if (g[c[0], c[1]] == g[nbr[0], nbr[1]]):
                    candidates = np.append(candidates, [nbr], axis=0)

    return cluster


def energy(g):
    nrg = 0
    for x in range(N):
        for y in range(N):
            nrg -= g[x, y] * g[(x + 1) % N, y]
            nrg -= g[x, y] * g[x, (y + 1) % N]
    return nrg / (N ** 2)


# Initialize grid of spins
grid = np.random.choice(choices, [N, N])

# Keep track of average magnitization as a diagnostic
# for reaching thermal equilibrium
m = [np.average(grid)]
e = [energy(grid)]

flips = 0

while flips < K * N * N:
    if progress:
        print("%4.1f" % (100 * flips / (K * N * N)), "%")

    # Get positions of spins in a cluster and flip all members
    clust = update(grid)
    for c in clust:
        grid[c[0], c[1]] *= -1

    m = np.append(m, np.average(grid))
    e = np.append(e, energy(grid))
    flips += len(clust)


if display or save:
    maj = m[-1]
    majSpins = np.empty([0, 2])
    for x in range(N):
        for y in range(N):
            if grid[x, y] * maj > 0:
                majSpins = np.append(majSpins, [[x, y]], axis=0)

if save:
    # Scan through the final spin configuration and save the locations
    # of those spins which are (anti-)aligned with the majority

    fileDir = os.path.dirname(os.path.realpath('__file__'))
    pathN = os.path.join(fileDir, 'Data_2d_Ising_Wolff_N=%s_K=%s' % (N, K))
    pathNT = os.path.join(fileDir, 'Data_2d_Ising_Wolff_N=%s_K=%s/%s' % (N, K, T))
    filename = os.path.join(fileDir, 'Data_2d_Ising_Wolff_N=%s_K=%s/%s/%s.txt'
                            % (N, K, T, int(time.time())))

    # Create folders if necessary
    if not os.path.exists(pathN):
        os.makedirs(pathN)
    if not os.path.exists(pathNT):
        os.makedirs(pathNT)

    np.savetxt(filename, majSpins, fmt='%d')


if display:
    fig, axes = pyplot.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(abs(m))
    axes[1].plot(e)
    pyplot.show()

    pyplot.figure(figsize=(5, 5))
    pyplot.scatter(majSpins[:, 0], majSpins[:, 1], marker='.', s=10, c='k')
    pyplot.axis('off')
    pyplot.show()
