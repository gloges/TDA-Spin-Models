import numpy as np
import matplotlib.pyplot as pyplot
import time

# 2d Ising spin model
# Spins sit at lattice sites of square lattice.

N = 75  # Size of grid is NxN (N^2 total spins)
T = 2.4   # Temperature
K = 20   # Iterate until K * (N^2) spins have been flipped in total

choices = [-1, 1]   # Values to choose from to initialize grid


# Returns randomized grid
def randomGrid():
    return np.random.choice(choices, [N, N])


# Returns a random lattice site
def randomSite():
    return np.random.randint(0, N, 2)


def update(g):
    x, y = randomSite()
    cluster = np.array([[x, y]])
    candidates = np.empty([0, 2], dtype=int)

    nbrs = (np.array([x, y]) + [[0, 1], [0, -1], [1, 0], [-1, 0]]) % N
    for nbr in nbrs:
        if (g[x, y] == g[nbr[0], nbr[1]]):
            candidates = np.append(candidates, [nbr], axis=0)

    while len(candidates) > 0:
        c = candidates[0]
        candidates = candidates[1:]
        if (cluster == c).all(1).any():
            continue
        elif (np.random.random() < 1 - np.exp(-2. / T)):
            cluster = np.append(cluster, [c], axis=0)
            nbrs = (c + [[0, 1], [0, -1], [1, 0], [-1, 0]]) % N
            for nbr in nbrs:
                if (g[c[0], c[1]] == g[nbr[0], nbr[1]]):
                    candidates = np.append(candidates, [nbr], axis=0)

    return cluster


# Initialize grid of spins
grid = randomGrid()

# Keep track of average magnitization as a diagnostic
# for reaching thermal equilibrium
m = [np.average(grid)]

flips = 0

while flips < K * N * N:
    print("%4.1f" % (100 * flips / (K * N * N)), "%")
    clust = update(grid)
    for c in clust:
        grid[c[0], c[1]] *= -1
    m = np.append(m, np.average(grid))
    flips += len(clust)


# maj = m[-1]
# toSave = np.empty([0, 2])
# for x in range(N):
#     for y in range(N):
#         if grid[x, y] * maj < 0:
#             toSave = np.append(toSave, [[x, y]], axis=0)

# np.savetxt(r'C:\Users\gerg1\Box Sync\TDA\Data_2d_Ising_Wolff_N=50\%s\%s.txt'
#            % (T, int(time.time())), toSave, fmt='%d')


fig, axes = pyplot.subplots(1, 2, figsize=(10, 4))
axes[0].plot(abs(m))
axes[1].matshow(grid)
axes[1].axis('off')
pyplot.show()
