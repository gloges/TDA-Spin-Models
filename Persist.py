import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.colors as colors
import gudhi
import os

N = 50
T = 1.9

fileDir = os.path.dirname(os.path.realpath('__file__'))
pathNT = os.path.join(fileDir, 'Data_2d_Ising_Wolff_N=%s\%s' % (N, T))

if not os.path.exists(pathNT):
    print("Folder does not exist: ", pathNT)
    quit()

files = os.listdir(pathNT)
data = [np.loadtxt("%s\%s" % (pathNT, f), dtype=int) for f in files]
print("Data imported:", len(data), "runs")

alphaComplexes = [gudhi.AlphaComplex(d) for d in data]
print("Alpha complexes constructed")

# Not sure what max_alpha_square controls
simplexTrees = [ac.create_simplex_tree(max_alpha_square=20)
                for ac in alphaComplexes]
print("Simplex trees created")

persistData = [st.persistence() for st in simplexTrees]
combinedPersistData = [a for b in persistData for a in b]
print("Persistence data computed")

pplot = gudhi.plot_persistence_diagram(combinedPersistData, alpha=0.01)
pplot.show()

cpd = np.array([np.asarray([a[0], a[1][0], a[1][1]])
                for a in combinedPersistData])

h1indices = (cpd[:, 0] == 1) & (np.isfinite(cpd[:, 2]))
h1Data = cpd[h1indices]

h1born = h1Data[:, 1]
h1death = h1Data[:, 2]

fig, axes = pyplot.subplots(2, 2, sharex=True, sharey='row', figsize=(10, 10))
pyplot.subplots_adjust(wspace=0, hspace=0)

axes[0, 0].scatter(h1born, h1death, alpha=0.01)
axes[0, 1].hist2d(h1born, h1death, bins=20,
                  norm=colors.SymLogNorm(linthresh=1))
axes[1, 0].scatter(h1born, h1death - h1born, alpha=0.01)
axes[1, 1].hist2d(h1born, h1death - h1born, bins=20,
                  norm=colors.SymLogNorm(linthresh=1))
pyplot.show()
