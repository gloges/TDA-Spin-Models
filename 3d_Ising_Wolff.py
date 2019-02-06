import numpy as np
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D

# 3d Ising spin model
# Spins sit at lattice sites of square lattice.

N = 20  # Size of grid is NxNxN (N^3 total spins)
T = 4.3   # Temperature
K = 15   # Iterate until K * (N^3) spins have been flipped in total

choices = [-1, 1]   # Values to choose from to initialize grid


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


# Initialize grid of spins
grid = randomGrid()

# Keep track of average magnitization as a diagnostic
# for reaching thermal equilibrium
m = [np.average(grid)]

flips = 0

while flips < K * N ** 3:
    print("%4.1f" % (100 * flips / (K * N ** 3)), "%")
    clust = update(grid)
    for c in clust:
        grid[c[0], c[1], c[2]] *= -1
    m = np.append(m, abs(np.average(grid)))
    flips += len(clust)

maj = np.average(grid)
toPlot = np.empty([0, 3])
for x in range(N):
    for y in range(N):
        for z in range(N):
            if maj * grid[x, y, z] < 0:
                toPlot = np.append(toPlot, [[x, y, z]], axis=0)

pyplot.plot(m)
fig = pyplot.figure()
ax3d = fig.add_subplot(111, projection='3d')
ax3d.scatter(toPlot[:, 0], toPlot[:, 1], toPlot[:, 2], s=1, c='k')
pyplot.show()
