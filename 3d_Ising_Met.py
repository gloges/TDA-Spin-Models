import numpy as np
import matplotlib.pyplot as pyplot

# 3d Ising spin model
# Spins sit at lattice sites of square lattice.

N = 20  # Size of grid is NxNxN (N^3 total spins)
T = 4.4   # Temperature
K = 1000   # Number of iterations is K * (N^3)

choices = [-1, 1]   # Values to choose from to initialize grid


# Returns randomized grid
def randomGrid():
    return np.random.choice(choices, [N, N, N])


# Returns a random lattice site
def randomSite():
    return np.random.randint(0, N, 3)


# Returns sum of six neighboring spins
def nbrProduct(g, x, y, z):
    s1 = g[x % N, y % N, (z + 1) % N]
    s2 = g[x % N, y % N, (z - 1) % N]
    s3 = g[x % N, (y + 1) % N, z % N]
    s4 = g[x % N, (y - 1) % N, z % N]
    s5 = g[(x + 1) % N, y % N, z % N]
    s6 = g[(x - 1) % N, y % N, z % N]
    return s1 + s2 + s3 + s4 + s5 + s6


# Change in energy from flipping spin at (x,y,z)
def deltaE(g, x, y, z):
    return 2 * g[x, y, z] * nbrProduct(g, x, y, z)


# Determine if spin at (x,y,z) should flip
def flipSpinB(g, x, y, z):
    dE = deltaE(g, x, y, z)
    if np.random.random() < np.exp(-dE / T):
        return -1
    else:
        return 1


# Choose N^3 random spins (on average each is chosen once)
# and decide to flip or not
def wash(g):
    gNew = g
    for a in range(N ** 3):
        x, y, z = randomSite()
        gNew[x, y, z] *= flipSpinB(gNew, x, y, z)
    return gNew


# Initialize grid of spins
grid = randomGrid()

# Keep track of average magnitization as a diagnostic
# for reaching thermal equilibrium
m = [np.average(grid)]

# Wash the grid K times
for k in range(K):
    print("%4.1f" % (100 * k / K), "%")
    grid = wash(grid)
    m = np.append(m, np.average(grid))


# Plot the magnitization as a function of number of washes,
# and display the final spin configuration
fig, axes = pyplot.subplots(1, 2, figsize=(10, 4))
axes[0].plot(m)
axes[1].matshow(grid[0])    # Just show a slice of lattice
axes[1].axis('off')
pyplot.show()
