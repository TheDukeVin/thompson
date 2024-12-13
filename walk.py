
import numpy as np
import matplotlib.pyplot as plt

p = 0.5
M = 1e+18

s = 0
n = 0
hist = []
while n < M:
    inc = max(1, int(n/100))
    n += inc
    s += np.random.binomial(inc, p)
    hist.append([n, s])

hist = np.array(hist)

fig, ax = plt.subplots()
ax.plot(hist[:, 0], np.sqrt(hist[:, 0]) * (hist[:, 1] / hist[:, 0] - p) / np.sqrt(p*(1-p)))
ax.set_xscale('log')

ax.plot([1, M], [0, 0])

plt.show()