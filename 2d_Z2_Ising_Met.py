import numpy as np
#import matplotlib.pyplot as pyplot
import os
import sys
import time

# 2d, Z2 gauge Ising spin model
# Lattice sites form a square grid and are labelled by 0<=x,y<N
# Spins lie on edges connecting adjacent lattice sites and are
# labelled by (x,y,a):
#   - 0<=x,y<N give the corresponding lattice site.
#   - a=0,1 respectively label the horizonal edge to the right
#       of (x,y) and the vertical edge above (x,y).
# Each plaquette is labelled by its lower-right lattice site.
# See "https://www.itp3.uni-stuttgart.de/downloads/
#         Lattice_gauge_theory_SS_2009/Chapter3.pdf" for a discussion.

N = 50  # Size of grid is NxN (2*N^2 total spins)

# T = 0.7   # Temperature
T = float(sys.argv[1]) / float(sys.argv[2])   # Read in temp from cp args

K = 200   # Average number of flips per spin

choices = [-1, 1]   # Values to choose from to initialize grid

display = False
save = True
progress = True


# Returns randomized grid
def randomGrid():
    return np.random.choice(choices, [N, N, 2])


# Returns a random lattice site (not a spin site)
def randomSite():
    return np.random.randint(0, N, 2)


# Returns product of four spins on the plaquette labelled by (x,y)
def plaqProduct(g, x, y):
    s1 = g[x % N, y % N, 0]
    s2 = g[x % N, y % N, 1]
    s3 = g[(x + 1) % N, y % N, 1]
    s4 = g[x % N, (y + 1) % N, 0]
    return s1 * s2 * s3 * s4


# Returns energy of spin configuration, g
def energy(g):
    nrg = 0
    for x in range(N):
        for y in range(N):
            nrg -= plaqProduct(g, x, y)
    return nrg / (2 * N ** 2)


# Change in energy from flipping spin at (x,y,a)
def deltaE(g, x, y, a):
    dE = 2 * plaqProduct(g, x, y)
    if a == 0:
        dE += 2 * plaqProduct(g, x, y - 1)
    else:
        dE += 2 * plaqProduct(g, x - 1, y)
    return dE


# Determine if spin at (x,y,a) should flip
def flipSpinB(g, x, y, a):
    dE = deltaE(g, x, y, a)
    if np.random.random() < np.exp(-dE / T):
        return -1
    else:
        return 1


# Choose 2*N^2 random spins (on average each is chosen once)
# and decide to flip or not
def wash(g):
    gNew = g
    for z in range(2 * N ** 2):
        x, y = randomSite()
        a = np.random.choice(2)
        gNew[x, y, a] *= flipSpinB(gNew, x, y, a)
    return gNew


# Returns positions of spins aligned with majority
def majSpinPosn(g):
    maj = np.average(g)
    majPosn = np.empty([0, 2])

    for x in range(N):
        for y in range(N):
            if g[x, y, 0] * maj > 0:
                majPosn = np.append(majPosn, [[x + 0.5, y]], axis=0)
            if g[x, y, 1] * maj > 0:
                majPosn = np.append(majPosn, [[x, y + 0.5]], axis=0)

    return majPosn

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


    if save:
        # Scan through the final spin configuration and save the locations
        # of those spins which are (anti-)aligned with the majority
        toSave = majSpinPosn(grid)

        fileDir = os.path.dirname(os.path.realpath('__file__'))
        pathN = os.path.join(fileDir, 'Data_2d_Z2_Ising_Met_N=%s_K=%s' % (N, K))
        pathNT = os.path.join(fileDir, 'Data_2d_Z2_Ising_Met_N=%s_K=%s/%s' % (N, K, T))
        filename = os.path.join(fileDir, 'Data_2d_Z2_Ising_Met_N=%s_K=%s/%s/%s.txt'
                                % (N, K, T, int(time.time())))

        # Create folders if necessary
        if not os.path.exists(pathN):
            os.makedirs(pathN)
        if not os.path.exists(pathNT):
            os.makedirs(pathNT)

        np.savetxt(filename, toSave, fmt='%.1f')


'''if display:
    # Plot the magnitization as a function of number of washes,
    # and display the final spin configuration
    majPosn = majSpinPosn(grid)

    fig, axes = pyplot.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(m)
    axes[1].plot(e)
    pyplot.show()

    pyplot.figure(figsize=(5, 5))
    pyplot.scatter(majPosn[:, 0], majPosn[:, 1], marker='.', s=10, c='k')
    pyplot.axis('off')
    pyplot.show()'''
