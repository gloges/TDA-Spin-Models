import numpy as np
import matplotlib.pyplot as pyplot
from matplotlib.colors import LogNorm
from ripser import ripser, plot_dgms
from scipy.stats import kde
import os

T = 1.4

loc = r'C:\Users\gerg1\Box Sync\TDA\Data_2d_Ising_Wolff_N=50\%s' % T
files = os.listdir(loc)
data = [np.loadtxt("%s\%s" % (loc, f), dtype=int) for f in files]


h1s = [(ripser(d)['dgms'])[1] for d in data]

born1s = np.empty(0)
death1s = np.empty(0)
for i in range(len(h1s)):
    born1s = np.append(born1s, h1s[i][:, 0])
    death1s = np.append(death1s, h1s[i][:, 1])

life1s = death1s - born1s

# plot_dgms(dgms, lifetime=True, show=True)

k = kde.gaussian_kde([born1s, life1s])
xi, yi = np.mgrid[born1s.min():born1s.max():100 * 1j,
                  life1s.min():life1s.max():100 * 1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

fig, axes = pyplot.subplots(1, 2, sharey=True, figsize=(12, 5))

axes[0].hist2d(born1s, life1s, norm=LogNorm())
axes[1].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud')
pyplot.show()
