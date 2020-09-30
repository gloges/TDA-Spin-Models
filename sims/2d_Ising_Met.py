import numpy as np
import matplotlib.pyplot as pyplot

# 2d Ising spin model
# Spins sit at lattice sites of square lattice.

N = 50  # Size of grid is NxN (N^2 total spins)
T = 1.4   # Temperature
K = 100   # Number of iterations is K * (N^2)

choices = [-1, 1, 1]   # Values to choose from to initialize grid


# Returns randomized grid
def randomGrid():
    return np.random.choice(choices, [N, N])


# Returns a random lattice site
def randomSite():
    return np.random.randint(0, N, 2)


# Returns sum of four neighboring spins
def nbrSum(g, x, y):
    s1 = g[x % N, (y + 1) % N]
    s2 = g[x % N, (y - 1) % N]
    s3 = g[(x + 1) % N, y % N]
    s4 = g[(x - 1) % N, y % N]
    return s1 + s2 + s3 + s4


# Change in energy from flipping spin at (x,y)
def deltaE(g, x, y):
    return 2 * g[x, y] * nbrSum(g, x, y)


# Determine if spin at (x,y) should flip
def flipSpinB(g, x, y):
    dE = deltaE(g, x, y)
    if np.random.random() < np.exp(-dE / T):
        return -1
    else:
        return 1


# Choose N^2 random spins (on average each is chosen once)
# and decide to flip or not
def wash(g):
    gNew = g
    for a in range(N ** 2):
        x, y = randomSite()
        gNew[x, y] *= flipSpinB(gNew, x, y)
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
axes[1].matshow(grid)
axes[1].axis('off')
pyplot.show()
