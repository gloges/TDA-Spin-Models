import numpy as np
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import time

# 3d, Z2 gauge Ising spin model
# Lattice sites form a square grid and are labelled by 0<=x,y,z<N
# Spins lie on edges connecting adjacent lattice sites and are
# labelled by (x,y,z,a):
#   - 0<=x,y,z<N give the corresponding lattice site.
#   - a=0,1,2 gives the direction of the edge from the lattice
#       site on which the spin lies
# Each plaquette is labelled by its "lowest" lattice site and
# by a,b=0,1,2 where a!=b correspond to the x,y,z-directions.

N = 15  # Size of grid is NxNxN (3*N^3 total spins)

T = 0.01   # Temperature
# T = float(sys.argv[1]) / float(sys.argv[2])   # Read in temp from cp args

p = 0.2     # Probability for choosing bond weights +/-1

K = 200   # Number of iterations is K * (3*N^3)

choices = [-1, 1]   # Values to choose from to initialize grid

display = True
save = False
progress = True


# Returns randomized grid
def randomGrid():
    return np.random.choice(choices, [N, N, N])


# Returns random bonds
def randomBonds():
    randNums = np.random.rand(N, N, N, 3)
    Js = np.floor(randNums + p)     # Sets to either zero or one
    Js = 2 * Js - 1
    return Js


# Returns a random lattice site
def randomSite():
    return np.random.randint(0, N, 3)


# Returns sum of spins at vertex (x,y,z)
def vertSum(g, x, y, z):
    sx1 = Jlist[(x - 1) % N, y, z, 0] * g[(x - 1) % N, y % N, z % N]
    sx2 = Jlist[x, y, z, 0] * g[(x + 1) % N, y % N, z % N]
    sy1 = Jlist[x, (y - 1) % N, z, 1] * g[x % N, (y - 1) % N, z % N]
    sy2 = Jlist[x, y % N, z, 1] * g[x % N, (y + 1) % N, z % N]
    sz1 = Jlist[x, y, (z - 1) % N, 2] * g[x % N, y % N, (z - 1) % N]
    sz2 = Jlist[x, y, z, 2] * g[x % N, y % N, (z + 1) % N]

    return sx1 + sx2 + sy1 + sy2 + sz1 + sz2


# Returns energy of spin configuration, g
def energy(g):
    return 0


# Change in energy from flipping spin at (x,y,z)
def deltaE(g, x, y, z):
    return 2 * g[x, y, z] * vertSum(g, x, y, z)


# Determine if spin at (x,y,z) should flip
def flipSpinB(g, x, y, z):
    dE = deltaE(g, x, y, z)
    if np.log(np.random.random()) < -dE / T:
        return -1
    else:
        return 1


# Choose N^3 random spins (on average each is chosen once)
# and decide to flip or not
def wash(g):
    gNew = g
    for z in range(N ** 3):
        x, y, z = randomSite()
        gNew[x, y, z] *= flipSpinB(gNew, x, y, z)
    return gNew


# Initialize bonds to +/-1
Jlist = randomBonds()


for i in np.arange(5):
    # Initialize grid of spins
    grid = randomGrid()

    # Keep track of average magnitization as a diagnostic
    # for reaching thermal equilibrium
    m = [np.average(grid)]
    e = [energy(grid)]

    # Wash the grid K times
    for k in range(K):
        if progress:
            print("%4.1f" % (100 * k / K), "%")
        grid = wash(grid)
        m = np.append(m, np.average(grid))
        e = np.append(e, energy(grid))

    if save or display:
        maj = m[-1]
        majLoc = np.empty([0, 3])
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    if grid[x, y, z] * maj > 0:
                        majLoc = np.append(majLoc, [[x, y, z]], axis=0)

    if save:
        # Scan through the final spin configuration and save the locations
        # of those spins which are (anti-)aligned with the majority
        fileDir = os.path.dirname(os.path.realpath('__file__'))
        pathN = os.path.join(fileDir, 'Data_3d_glass_Met_N=%s_K=%s' % (N, K))
        pathNT = os.path.join(fileDir, 'Data_3d_glass_Met_N=%s_K=%s/%s' % (N, K, T))
        filename = os.path.join(fileDir, 'Data_3d_glass_Met_N=%s_K=%s/%s/%s.txt'
                                % (N, K, T, int(time.time())))

        # Create folders if necessary
        if not os.path.exists(pathN):
            os.makedirs(pathN)
        if not os.path.exists(pathNT):
            os.makedirs(pathNT)

        np.savetxt(filename, majLoc, fmt='%.1f')

    if display:
        # fig = pyplot.figure(figsize=(15, 5))
        # ax0 = fig.add_subplot(1, 3, 1)
        # ax0.plot(m)
        # ax1 = fig.add_subplot(1, 3, 2)
        # ax1.plot(e)
        # ax2 = fig.add_subplot(1, 3, 3, projection='3d')
        # ax2.scatter(majLoc[:, 0], majLoc[:, 1], majLoc[:, 2], s=1, c='k')

        pts1 = np.empty([0, 2])
        pts2 = np.empty([0, 2])
        for s in majLoc:
            if s[2] == 1:
                pts1 = np.append(pts1, [[s[0], s[1]]], axis=0)
            if s[2] == 8:
                pts2 = np.append(pts2, [[s[0], s[1]]], axis=0)
        fig = pyplot.figure(figsize=(10, 5))
        ax0 = fig.add_subplot(1, 2, 1)
        ax0.scatter(pts1[:, 0], pts1[:, 1], marker='o', s=15, c='k')
        ax1 = fig.add_subplot(1, 2, 2)
        ax1.scatter(pts2[:, 0], pts2[:, 1], marker='o', s=15, c='k')
        # pyplot.axis('off')
        pyplot.show()
