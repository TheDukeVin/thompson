
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.stats import norm
from scipy.stats import geom
import time

badArm = 0.3
fixedArm = 0.5

fig, ax = plt.subplots()

for trial in range(20):
    numSuccess = 0
    numPulls = 0

    t = 0
    timestamps = []

    while t < 1e+15:
        pullProb = 1 - beta.cdf(fixedArm, numSuccess + 1, numPulls - numSuccess + 1)
        pullTime = geom.rvs(pullProb)
        t += pullTime
        timestamps.append(t)
        numPulls += 1
        numSuccess += np.random.uniform() < badArm

    ax.set_xscale('log')
    ax.plot(timestamps, np.arange(len(timestamps)))
plt.show()