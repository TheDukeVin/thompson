
# Plot posterior arm means/variances as well.
# Plot trajectories of mean of arm means, variance of arm means.
# Five rows of plots, one for each statistic. Four different priors. For each prior,
# Draw 1000 trajectories, make histogram at different points in time (1, 10, 100, 1000).
# If histograms look like bell-shpaed, can collapse histogram to only mean/variance. Then can plot this over time.

# Plot cdfs of posterior means, three line types by checkpoint, five line colors by prior mean.
# Try looking at medians of posterior means instead of mean of means.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.stats import norm
from scipy.stats import geom
from scipy.stats import binom
from scipy.stats import ecdf
import time

class BetaTwoArmSim:
    def __init__(self):
        self.reset()
    
    def reset(self):

        self.armProbs = np.array([0.5, 0.5])

        self.numPulls = np.array([0, 0])
        self.numSuccess = np.array([0, 0])

    def onePull(self):
        samples = np.random.beta(self.numSuccess + 1, self.numPulls - self.numSuccess + 1)
        pulls = np.array([0, 0])
        pulls[np.argmax(samples)] = 1
        return pulls

    def asymptoticProbability(self): # Probability of pulling arm 0.
        alpha = self.numSuccess + 1
        beta = self.numPulls - self.numSuccess + 1
        mean = alpha / (alpha + beta)
        sd = np.sqrt(alpha)*np.sqrt(beta)/((alpha + beta) * np.sqrt(alpha + beta + 1))

        return norm.cdf((mean[0] - mean[1]) / (np.sqrt(sd[0]*sd[0] + sd[1]*sd[1])))

    def getPulls(self, totalPulls):
        if totalPulls < 5:
            s = np.array([0, 0])
            for i in range(0, totalPulls):
                s += self.onePull()
            return s
        prob = self.asymptoticProbability()
        arm0Pulls = binom.rvs(totalPulls, prob)
        return np.array([arm0Pulls, totalPulls-arm0Pulls])

    def sim(self, ax):
        totalPulls = 0
        hist = []
        while totalPulls < 1e+15:
            newPulls = max(1, int(totalPulls/100))
            totalPulls += newPulls
            pulls = self.getPulls(newPulls)
            self.numPulls += pulls
            self.numSuccess += binom.rvs(n=pulls, p=self.armProbs)
            hist.append(np.copy(self.numPulls))

        hist = np.array(hist)
        L, _ = hist.shape

        ax.plot(hist[:, 0] + hist[:, 1], hist[:, 0] / (hist[:, 0] + hist[:, 1]))

class NormalTwoArmSim:
    def __init__(self):
        self.reset()
    
    def reset(self):

        self.armMeans = np.array([0.0, 0.0])

        self.numPulls = np.array([0, 0])
        self.sumVals = np.array([0.0, 0.0])

        self.priorMean = np.array([0, 0])
        self.priorVariance = np.array([1, 1])

        self.sigma2 = 1.0
    
    def getPosterior(self):
        posteriorVariance = 1/(self.numPulls / self.sigma2 + 1 / self.priorVariance)
        posteriorMean = posteriorVariance * (self.sumVals / self.sigma2 + self.priorMean / self.priorVariance)
        return posteriorMean, posteriorVariance

    def asymptoticProbability(self): # Probability of pulling arm 0.
        posteriorMean, posteriorVariance = self.getPosterior()
        return norm.cdf((posteriorMean[0] - posteriorMean[1]) / (np.sqrt(posteriorVariance[0] + posteriorVariance[1])))

    def getPulls(self, totalPulls):
        prob = self.asymptoticProbability()
        arm0Pulls = binom.rvs(totalPulls, prob)
        return prob, np.array([arm0Pulls, totalPulls-arm0Pulls])
    
    def one_step_sim(self, Horizon, ax=None, color=None):
        hist = []
        for i in range(Horizon):
            hist.append(np.concatenate(self.getPosterior()))
            prob, pulls = self.getPulls(1)
            self.numPulls += pulls

            self.sumVals += norm.rvs(loc=self.armMeans * pulls, scale=np.sqrt(self.sigma2 * pulls))
        
        hist = np.array(hist)
        L, _ = hist.shape

        # plot_quantity = hist[:, 0] / (hist[:, 0] + hist[:, 1])
        plot_quantity = hist[:, 0]
        if ax is not None:
            count = np.arange(1, Horizon+1)
            ax.plot(count, plot_quantity * np.sqrt(count), color=color)
        return hist


    def sim(self, ax=None):
        totalPulls = 0
        hist = []
        while totalPulls < 1e+15:
            newPulls = max(1, int(totalPulls/1000))
            totalPulls += newPulls
            prob, pulls = self.getPulls(newPulls)
            self.numPulls += pulls
            hist.append(np.array([self.numPulls[0], self.numPulls[1], prob]))

            self.sumVals += norm.rvs(loc=self.armMeans * pulls, scale=np.sqrt(self.sigma2 * pulls))

        hist = np.array(hist)
        L, _ = hist.shape

        # plot_quantity = hist[:, 0] / (hist[:, 0] + hist[:, 1])
        plot_quantity = hist[:, 2]
        if ax is not None:
            ax.plot(hist[:, 0] + hist[:, 1], plot_quantity)
        return plot_quantity

# x = BetaTwoArmSim()
x = NormalTwoArmSim()

print("STARTING SIMULATION...")
start_time = time.time()

trials = [
    (-2, "orange"),
    (-1, "blue"),
    (0, "red"),
    (1, "green"),
    (2, "pink")
]

mode = "AGGREGATE"

if mode == "CHECKPOINT":
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    # for (mean, color) in trials:
    #     for i in range(10):
    #         x.reset()
    #         x.priorMean[0] = mean
    #         x.one_step_sim(ax, color)
    # plt.savefig("prior_dependence")



    # Plot distribution of normalized arm means

    all_means = []
    all_vars = []
    checkpoints = np.array([1, 10, 100, 1000])
    numReplicates = 1000

    mode = "READ"

    if mode == "WRITE":
        for t, (mean, color) in enumerate(trials):
            means = []
            vars = []
            for i in range(numReplicates):
                x.reset()
                x.priorMean[0] = mean
                nums = x.one_step_sim(1000)
                means.append(nums[checkpoints-1, 0])
                vars.append(nums[checkpoints-1, 2])
            all_means.append(np.array(means))
            all_vars.append(np.array(vars))
        all_means = np.array(all_means)
        all_vars = np.array(all_vars)
        np.savetxt("means.csv", all_means.flatten())
        np.savetxt("vars.csv", all_vars.flatten())
    elif mode == "READ":
        all_means = np.loadtxt("means.csv").reshape((len(trials), numReplicates, len(checkpoints)))
        all_vars = np.loadtxt("vars.csv").reshape((len(trials), numReplicates, len(checkpoints)))

    fig, ax = plt.subplots(5, 4, figsize=(18, 18))
    for t, (mean, color) in enumerate(trials):
        for i in range(4):
            ax[t][i].hist(all_means[t,:,i]*np.sqrt(checkpoints[i]), color=color, bins=np.arange(np.min(all_means[t,:,i]*np.sqrt(checkpoints[i])), 8, 0.2))
    plt.savefig("prior_dependence_means")
    plt.tight_layout()
    fig, ax = plt.subplots(5, 4, figsize=(18, 18))
    for t, (mean, color) in enumerate(trials):
        for i in range(4):
            ax[t][i].hist(all_vars[t,:,i]*checkpoints[i], color=color, bins=np.arange(0, 20, 0.2))
    plt.savefig("prior_dependence_vars")
    plt.tight_layout()

if mode == "AGGREGATE":
    numReplicates = 10
    Horizon = 1000

    all_aggs = []

    mode = "WRITE"

    if mode == "WRITE":
        for t, (mean, color) in enumerate(trials):
            agg = np.zeros((Horizon, 4))
            for i in range(numReplicates):
                x.reset()
                x.priorMean[0] = mean
                nums = x.one_step_sim(Horizon)
                agg += nums
            all_aggs.append(agg / numReplicates)
        all_aggs = np.array(all_aggs)
        np.savetxt("agg/agg.csv", all_aggs.flatten())
    elif mode == "READ":
        all_aggs = np.loadtxt("agg/agg.csv").reshape((len(trials), Horizon, 4))
    
    for i in range(4):
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        count = np.arange(1, Horizon+1)
        if i<2:
            scale = np.sqrt(count)
            kind = 'mean'
        else:
            scale = count
            kind = 'var'
        for t, (mean, color) in enumerate(trials):
            ax.plot(count, all_aggs[t, :, i]*scale, color=color)
        plt.savefig(f"agg/{kind}{i%2}")


# ECDF code
# res = ecdf(np.array(all_nums))
# res.cdf.plot(ax, color=color)
# plt.savefig("prior_dependence_ecdf")

if mode == "LONG_SIM":
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    # ax.set_yscale('log')

    for i in range(10):
        x.reset()
        x.sim(ax)
    plt.savefig("zero_margin_ts")

    all_nums = np.array([])

    for i in range(100):
        x.reset()
        nums = x.sim(ax)
        all_nums = np.concatenate((all_nums, nums))

    fig, ax = plt.subplots()
    plt.hist(all_nums, bins=30)
    plt.savefig("zero_margin_hist")

print("TIME: " + str(time.time() - start_time))