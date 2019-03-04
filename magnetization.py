import numpy as np
import matplotlib.pyplot as pyplot
import os


N = 50

magData = np.empty([0, 2])


def M_Onsag(T):
    if T > 2.269:
        return 0
    else:
        return (1 - 1 / np.sinh(2 / T) ** 4) ** (1 / 8)


# Locate folder of spin data
fileDir = os.path.dirname(os.path.realpath('__file__'))

for i in np.arange(200, 240, 1):
    T = i / 100
    print(T)
    pathNT = os.path.join(fileDir, 'Data_2d_Ising_Wolff_N=%s\%s' % (N, T))
    if not os.path.exists(pathNT):
        print("Folder does not exist:", pathNT)
        continue

    # Load spin config data from folder
    files = os.listdir(pathNT)
    for f in files:
        if f.endswith(".txt"):
            maj = len(np.loadtxt("%s\%s" % (pathNT, f), dtype=int))
            m = 2 * maj / (N ** 2) - 1
            magData = np.append(magData, [[T, m]], axis=0)


tRange = np.linspace(0, 4, 500)
mExact = [M_Onsag(t) for t in tRange]

pyplot.plot(tRange, mExact, c='r', zorder=0)
pyplot.scatter(magData[:, 0], magData[:, 1], alpha=0.02, c='k', zorder=1)
pyplot.show()
