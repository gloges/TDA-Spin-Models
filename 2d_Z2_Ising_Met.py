import numpy as np
import matplotlib.pyplot as pyplot
import os
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
T = 1.0   # Temperature
K = 500   # Average number of flips per spin

choices = [-1, 1]   # Values to choose from to initialize grid

display = False
save = True
progress = False


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
            nrg += plaqProduct(g, x, y)
    return nrg / (2 * N ** 2)


# Change in energy from flipping spin at (x,y,a)
def deltaE(g, x, y, a):
    dE = 2 * plaqProduct(g, x, y)
    if a == 0:
        dE *= plaqProduct(g, x, y - 1)
    else:
        dE *= plaqProduct(g, x - 1, y)
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


# Gives value of spin closest to (x,y)
# (For displaying purposes)
def G(g, x, y):
    xInt = int(np.floor(x))
    yInt = int(np.floor(y))
    xFrac = x - xInt
    yFrac = y - yInt

    if yFrac > xFrac:
        xA = xInt
        if yFrac < 1 - xFrac:
            yA = yInt
            a = 1
        else:
            yA = yInt + 1
            a = 0
    else:
        yA = yInt
        if yFrac <= 1 - xFrac:
            xA = xInt
            a = 0
        else:
            xA = xInt + 1
            a = 1
    return g[xA % N, yA % N, a]


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

avenrg = [np.mean(e[:(i + 1)]) for i in range(len(e))]


if save:
    # Scan through the final spin configuration and save the locations
    # of those spins which are (anti-)aligned with the majority
    maj = m[-1]
    toSave = np.empty([0, 2])
    for x in range(N):
        for y in range(N):
            if grid[x, y, 0] * maj > 0:
                toSave = np.append(toSave, [[x + 0.5, y]], axis=0)
            if grid[x, y, 1] * maj > 0:
                toSave = np.append(toSave, [[x, y + 0.5]], axis=0)

    fileDir = os.path.dirname(os.path.realpath('__file__'))
    pathN = os.path.join(fileDir, 'Data_2d_Z2_Ising_Met_N=%s' % N)
    pathNT = os.path.join(fileDir, 'Data_2d_Z2_Ising_Met_N=%s/%s' % (N, T))
    filename = os.path.join(fileDir, 'Data_2d_Z2_Ising_Met_N=%s/%s/%s.txt'
                            % (N, T, int(time.time())))

    # Create folders if necessary
    if not os.path.exists(pathN):
        os.makedirs(pathN)
    if not os.path.exists(pathNT):
        os.makedirs(pathNT)

    np.savetxt(filename, toSave, fmt='%.1f')


if display:
    # Plot the magnitization as a function of number of washes,
    # and display the final spin configuration
    xRange = np.linspace(0, N, 10 * N)
    yRange = np.linspace(0, N, 10 * N)
    zValues = [[G(grid, x, y) for x in xRange] for y in yRange]

    fig, axes = pyplot.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(m)
    axes[1].plot(avenrg)
    axes[2].imshow(zValues)
    axes[2].axis('off')
    pyplot.show()
