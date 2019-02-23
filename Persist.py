import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.colors as colors
import gudhi
import os
import time

N = 50    # Hard-coded parameters to specify the folder of spin
T = 2.27	  # configurations to run through (2d Wolff)

# Locate folder of spin data
fileDir = os.path.dirname(os.path.realpath('__file__'))
pathNT = os.path.join(fileDir, 'Data_2d_Ising_Wolff_N=%s\%s' % (N, T))
if not os.path.exists(pathNT):
    print("Folder does not exist:", pathNT)
    quit()

# Load spin config data from folder
files = os.listdir(pathNT)
data = [np.loadtxt("%s\%s" % (pathNT, f), dtype=int) for f in files]
print("Data imported:", len(data), "runs")

alphaComplexes = [gudhi.AlphaComplex(d) for d in data]
print("Alpha complexes constructed")

# Appropriate value of α^2? Needs to be smaller than N to get
# the speed-up, but large enough to capture all of the interesting stuff...
# Leave unspecified?
simplexTrees = [ac.create_simplex_tree()
                for ac in alphaComplexes]
print("Simplex trees created")

persistData = [st.persistence() for st in simplexTrees]
combinedPersistData = [a for b in persistData for a in b]
print("Persistence data computed")


# Convert to Numpy array
cpd = np.array([np.asarray([a[0], a[1][0], a[1][1]])
                for a in combinedPersistData])

# Extract H1 data for which the lifetime is not infinite
h1indices = (cpd[:, 0] == 1) & (np.isfinite(cpd[:, 2]))
h1Data = np.round(cpd[h1indices, 1:], decimals=5)

h1born = h1Data[:, 0]
h1death = h1Data[:, 1]


# Example run for comparison with cumulative
rdRun = np.array([np.asarray([a[0], a[1][0], a[1][1]]) for a in persistData[np.random.random_integers(0, len(persistData) - 1)]])
h1runindices = (rdRun[:, 0] == 1) & (np.isfinite(rdRun[:, 2]))
h1runData = np.round(rdRun[h1runindices, 1:], decimals=5)
h1runborn = h1runData[:, 0]
h1rundeath = h1runData[:, 1]

# # More compact cumulative data
# h1borndeath, counts = np.unique(h1Data, axis=0, return_counts=True)
# toSave = np.array([np.array([h1borndeath[i, 0], h1borndeath[i, 1], counts[i]])
#                    for i in range(len(h1borndeath))])


# Traditional persistance diagram
# pplot = gudhi.plot_persistence_diagram(combinedPersistData, alpha=0.01)
# pplot.show()

# pplot = gudhi.plot_persistence_diagram(persistData[0], alpha=0.01)
# pplot.show()


# Scatter and binned density plots
# Bins now are on-the-fly. Will need to be more systematic in binning.
fig, axes = pyplot.subplots(2, 4, sharex=True, sharey='row', figsize=(18, 9))
pyplot.subplots_adjust(wspace=0.05, hspace=0.05)
fig.suptitle("T = %.2f, %s runs" % (T, len(data)))

axes[0, 0].scatter(h1born, h1death, alpha=0.01)
axes[0, 0].set_ylabel('Death')
axes[0, 1].hist2d(h1born, h1death, bins=25, range=[[0, 25], [0, 25]],
                  norm=colors.SymLogNorm(linthresh=1))
axes[1, 0].scatter(h1born, h1death - h1born, alpha=0.01)
axes[1, 0].set_ylabel('Lifetime')
axes[1, 0].set_xlabel('Birth')
axes[1, 1].hist2d(h1born, h1death - h1born, bins=25, range=[[0, 25], [0, 25]],
                  norm=colors.SymLogNorm(linthresh=1))
axes[1, 1].set_xlabel('Birth')

axes[0, 2].scatter(h1runborn, h1rundeath, alpha=0.01)
axes[0, 3].hist2d(h1runborn, h1rundeath, bins=25, range=[[0, 25], [0, 25]],
                  norm=colors.SymLogNorm(linthresh=1))
axes[1, 2].scatter(h1runborn, h1rundeath - h1runborn, alpha=0.01)
axes[1, 2].set_xlabel('Birth')
axes[1, 3].hist2d(h1runborn, h1rundeath - h1runborn, bins=25, range=[[0, 25], [0, 25]],
                  norm=colors.SymLogNorm(linthresh=1))
axes[1, 3].set_xlabel('Birth')
pyplot.show()
