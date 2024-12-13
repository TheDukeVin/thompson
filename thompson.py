
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.stats import norm
import time
from test import *

class BernoulliBandit:
    def __init__(self, numArms, means="random"):
        self.numArms = numArms
        self.means = means
        self.random = False
        if type(self.means) is str and self.means == "random":
            self.random = True
        self.reset()
    
    def reset(self):
        self.numPulls = np.zeros(self.numArms)
        self.sumValue = np.zeros(self.numArms)
        self.pullHistory = []
        if self.random:
            self.means = np.random.uniform(size=self.numArms)
    
    def pull(self, arm):
        self.pullHistory.append(np.copy(self.numPulls))
        self.numPulls[arm] += 1
        self.sumValue[arm] += np.random.uniform() < self.means[arm]
    
    def Thompson(self):
        samples = np.random.beta(self.sumValue + 1, self.numPulls - self.sumValue + 1)
        return np.argmax(samples)
    
    def runThompson(self, numTrials):
        for i in range(numTrials):
            self.pull(self.Thompson())
    
    def getCredible(self, confidence):
        return np.array([beta.ppf((1-confidence)/2, self.sumValue + 1, self.numPulls - self.sumValue + 1),
                         beta.ppf((1+confidence)/2, self.sumValue + 1, self.numPulls - self.sumValue + 1)])
    
    def coverage(self, confidence):
        intervals = self.getCredible(confidence)
        return np.logical_and(intervals[0, :] < self.means, self.means < intervals[1, :])

def plot_pulls(bandit):
    colors = ['blue', 'orange']

    for t in range(10):
        bandit.reset()

        bandit.runThompson(10000)

        hist = np.array(bandit.pullHistory)

        for i in range(bandit.numArms):
            plt.plot(hist[:, i], color = colors[i])

    plt.show()

def calculate_coverage(bandit, numTrials=10000, episodeLength=1000):
    coverage = []
    numPulls = []
    for t in range(numTrials):
        bandit.reset()

        bandit.runThompson(episodeLength)

        numPulls.append(bandit.numPulls.copy())
        coverage.append(bandit.coverage(0.95))

    return np.array(coverage).mean(axis=0), np.array(numPulls).mean(axis=0)

if __name__ == '__main__':
    start_time = time.time()
    # plot_pulls(BernoulliBandit(numArms=2, means=[0.5, 0.5]))
    # plot_pulls(BernoulliBandit(numArms=2, means=[0.45, 0.5]))


    # TWO-armed bandits

    means = [[0.3, 0.5], [0.4, 0.5], [0.5, 0.5], [0.5, 0.6], [0.5, 0.7], "random"]
    numTrials = 10000

    mode = "READ"
    if mode == "WRITE":
        coverage = []
        counts = []
        for m in means:
            cov, count = calculate_coverage(BernoulliBandit(numArms=2, means=m), numTrials=numTrials)
            coverage.append(cov)
            counts.append(count)
        coverage = np.array(coverage)
        counts = np.array(counts)
        # coverage = np.array([calculate_coverage(BernoulliBandit(numArms=2, means=m)) for m in means])
        np.savetxt("coverage.csv", coverage)
        np.savetxt("counts.csv", counts)
    if mode == "READ":
        coverage = np.loadtxt("coverage.csv")
        counts = np.loadtxt("counts.csv")
    print("COVERAGE")
    print(coverage)
    print("COUNTS")
    print(counts)
    print(f"Time: {time.time() - start_time}")

    # Plot coverage
    species = [f"{a}/{b}" for a,b in means[:-1]] + ["random"]
    nums = {
        'Arm 1' : (coverage[:, 0], norm.ppf(0.975) * np.sqrt(coverage[:, 0]*(1-coverage[:, 0]) / numTrials)),
        'Arm 2' : (coverage[:, 1], norm.ppf(0.975) * np.sqrt(coverage[:, 1]*(1-coverage[:, 1]) / numTrials))
    }
    plot_grouped_bar_chart(species, nums, ylim=(0.9, 1))
    plt.hlines(0.95, -0.2, 5.5, color='red')
    plt.savefig("coverage")
    plt.show()