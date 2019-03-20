import numpy as np
#import matplotlib.pyplot as pyplot
#from mpl_toolkits.mplot3d import Axes3D
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

# T = 4.0   # Temperature
T = float(sys.argv[1]) / float(sys.argv[2])   # Read in temp from cp args

K = 200   # Number of iterations is K * (3*N^3)

choices = [-1, 1]   # Values to choose from to initialize grid

display = True
save = False
progress = True


# Returns randomized grid
def randomGrid():
    return np.random.choice(choices, [N, N, N, 3])


# Returns a random lattice site (not a spin site)
def randomSite():
    return np.random.randint(0, N, 3)


# Returns sum of spins at vertex (x,y,z)
def vertSum(g, x, y, z):
    s1 = g[x % N, y % N, z % N, 0]
    s2 = g[x % N, y % N, z % N, 1]
    s3 = g[x % N, y % N, z % N, 2]
    s4 = g[(x - 1) % N, y % N, z % N, 0]
    s5 = g[x % N, (y - 1) % N, z % N, 1]
    s6 = g[x % N, y % N, (z - 1) % N, 2]

    return s1 + s2 + s3 + s4 + s5 + s6


# Returns energy of spin configuration, g
def energy(g):
    nrg = 0
    for x in range(N):
        for y in range(N):
            for z in range(N):
                nrg += vertSum(g, x, y, z) ** 2
    return nrg / (3 * N ** 3)


# Change in energy from flipping spin at (x,y,z,a)
def deltaE(g, x, y, z, a):
    if a == 0:
        Ecurr = vertSum(g, x, y, z) ** 2 + vertSum(g, (x + 1) % N, y, z) ** 2
        Enew = (vertSum(g, x, y, z) - 2 * g[x, y, z, a]) ** 2 + (vertSum(g, (x + 1) % N, y, z) - 2 * g[x, y, z, a]) ** 2
    elif a == 1:
        Ecurr = vertSum(g, x, y, z) ** 2 + vertSum(g, x, (y + 1) % N, z) ** 2
        Enew = (vertSum(g, x, y, z) - 2 * g[x, y, z, a]) ** 2 + (vertSum(g, x, (y + 1) % N, z) - 2 * g[x, y, z, a]) ** 2
    else:
        Ecurr = vertSum(g, x, y, z) ** 2 + vertSum(g, x, y, (z + 1) % N) ** 2
        Enew = (vertSum(g, x, y, z) - 2 * g[x, y, z, a]) ** 2 + (vertSum(g, x, y, (z + 1) % N) - 2 * g[x, y, z, a]) ** 2
    return Enew - Ecurr


# Determine if spin at (x,y,z,a) should flip
def flipSpinB(g, x, y, z, a):
    dE = deltaE(g, x, y, z, a)
    if np.log(np.random.random()) < -dE / T:
        return -1
    else:
        return 1


# Choose 3*N^3 random spins (on average each is chosen once)
# and decide to flip or not
def wash(g):
    gNew = g
    for z in range(3 * N ** 3):
        x, y, z = randomSite()
        a = np.random.choice(3)
        gNew[x, y, z, a] *= flipSpinB(gNew, x, y, z, a)
    return gNew

for i in np.arange(1000):
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
                    if grid[x, y, z, 0] * maj > 0:
                        majLoc = np.append(majLoc, [[x + 0.5, y, z]], axis=0)
                    if grid[x, y, z, 1] * maj > 0:
                        majLoc = np.append(majLoc, [[x, y + 0.5, z]], axis=0)
                    if grid[x, y, z, 2] * maj > 0:
                        majLoc = np.append(majLoc, [[x, y, z + 0.5]], axis=0)

    if save:
        # Scan through the final spin configuration and save the locations
        # of those spins which are (anti-)aligned with the majority
        fileDir = os.path.dirname(os.path.realpath('__file__'))
        pathN = os.path.join(fileDir, 'Data_3d_squareice_Met_N=%s_K=%s' % (N, K))
        pathNT = os.path.join(fileDir, 'Data_3d_squareice_Met_N=%s_K=%s/%s' % (N, K, T))
        filename = os.path.join(fileDir, 'Data_3d_squareice_Met_N=%s_K=%s/%s/%s.txt'
                                % (N, K, T, int(time.time())))

        # Create folders if necessary
        if not os.path.exists(pathN):
            os.makedirs(pathN)
        if not os.path.exists(pathNT):
            os.makedirs(pathNT)

        np.savetxt(filename, majLoc, fmt='%.1f')


'''if display:
    fig = pyplot.figure(figsize=(15, 5))
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.plot(m)
    ax1 = fig.add_subplot(1, 3, 2)
    ax1.plot(e)
    ax2 = fig.add_subplot(1, 3, 3, projection='3d')
    ax2.scatter(majLoc[:, 0], majLoc[:, 1], majLoc[:, 2], s=1, c='k')
    pyplot.show()'''
