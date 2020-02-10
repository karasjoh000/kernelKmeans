import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import cycle
from scipy.stats import multivariate_normal


# performs kmean clustering given an array of two dim data.
class Kmeans(object):

    # initialize the object, pass in the data
    def __init__(self, data):
        self.data = data
        self.means = None
        self.assignment = np.zeros((self.data.shape[0], 1))
        self.k = None
        self.error = None

    # set the initial means to random data points
    def set_initial_means(self):
        self.means = self.data[np.random.choice(
            self.data.shape[0], self.k, replace=False), :]

    # calculate the distance between a data point and a mean
    # @staticmethod
    # def norm(x, y):
    #     distance = 0.0
    #     for i in range(0, x.shape[0]): 
    #         distance += (y[i] - x[i]) ** 2
    #     return math.sqrt(distance)

    # set k, number of clusters
    def set_k(self, k):
        self.k = k

    # perform the e step, assign data points to means that are closest to them.
    def e(self):
        for datum, index in zip(self.data, range(self.assignment.shape[0])):
            closest = (-1, math.inf)
            for mean, kth in zip(self.means, range(self.means.shape[0])):
                dist = np.dot(mean - datum, mean - datum)
                if dist < closest[1]:
                    closest = (kth, dist)
            self.assignment[index, 0] = closest[0]

    # recompute mean based on the dataum's assigned to it.
    def m(self):
        # (self.means.shape[0])
        for mean, kth in zip(self.means, range(self.means.shape[0])):
            # print("OLD: "+str(mean))
            count = 0.0
            mean.fill(0.0)
            for datum, assign in zip(self.data, self.assignment):
                if assign[0] == kth:
                    count += 1.0
                    for element, index in zip(datum, range(datum.shape[0])):
                        mean[index] += element
            self.means[kth, :] = np.divide(mean, count)
            # print("NEW: " + str(mean))

    # sum the errors of all the clusters
    def compute_error(self):
        error = 0.0
        for datum, assign in zip(self.data, self.assignment):
            error += np.dot(self.means[int(assign), :] -
                             datum, self.means[int(assign), :] - datum)
        return error

    # plot means and data without color coding
    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.data[:, 0], self.data[:, 1], "o", color="lightblue")
        ax.plot(self.means[:, 0], self.means[:, 1], "x", color="black")
        ax.set_title('kmeans')
        plt.show()

    # plot datapoints with color and save the graph
    def plot_color(self, iter):
        cycol = cycle('bgrcmk')
        fig, ax = plt.subplots()
        ax.plot(0, 0, "x", color="black")
        ax.set_title('kmeans ' + str(iter))
        for kth in range(self.means.shape[0]):
            col = next(cycol)
            for datum in range(self.data.shape[0]):
                if self.assignment[datum] == kth:
                    ax.plot(self.data[datum, 0],
                            self.data[datum, 1], ".", c=col,  alpha=0.7)
        ax.plot(self.means[:, 0], self.means[:, 1], "o", color="yellow")
        plt.show()
        name = "Kmeans" + str(iter) + ".pdf"
        fig.savefig(name, bbox_inches='tight')

    # perform em until the likelihood converges.
    def train(self):
        iter = 0
        error = math.inf
        while True:
            self.e()
            iter += 1
            self.m()
            new_error = self.compute_error()
            if new_error == error:
                break
            else:
                error = new_error
        self.error = error

    # run kmeans r times and return the object that had the smallest error.
    @staticmethod
    def run_r_times(r, k, data):
        best = None
        for i in range(r):
            new = Kmeans(data)
            new.set_k(k)
            new.set_initial_means()
            new.train()
            if best is None or best.error < new.error:
                best = new
        return best


# performs gmm em clustering
class Gmm(object):

    # initialize the object passing the data.
    def __init__(self, data, converge_epsilon=0.001):
        self.data = data
        self.mu = None
        self.sigma = None
        self.mix_coef = None
        self.k = None
        self.pdf = None
        self.prev_log = None
        self.assignments = None
        self.converge_epsilon = converge_epsilon

    # set the number of gaussian's to mix
    def set_k(self, k):
        self.k = k

    # perform kmeans clustering first to set the initial means, covariance matrices, and mix coefficients.
    def init_par(self):

        km = Kmeans(self.data)
        km.set_k(self.k)
        km.set_initial_means()
        km.train()
        self.mu = km.means  # initialize means from k means

        clusters = []
        # attach cluster labels to each datum
        ldata = np.hstack((km.data, km.assignment))
        for kth in range(km.means.shape[0]):
            clusters.append(
                ldata[ldata[:, 2] == kth][:, np.array([True, True, False])])  # organize data into their clusters
        clusters = np.array(clusters)  # convert list into an np array
        self.mix_coef = np.zeros((self.k, 1))
        self.sigma = np.zeros((self.k, 2, 2))
        for cluster, index in zip(clusters, range(
                clusters.shape[0])):  # for each cluster compute the mix coefficient and the within cluster covariance
            self.mix_coef[index, 0] = cluster.shape[0] / self.data.shape[0]
            self.sigma[index] = np.cov(cluster.transpose())

    # calculate the responsibility of datum x to a particular kth cluster.
    def calc_responsibility(self, kth, x):
        numerator = self.mix_coef[kth] * \
            multivariate_normal.pdf(x, self.mu[kth], self.sigma[kth])
        denominator = 0.0
        for cluster_num in range(self.k):
            denominator += self.mix_coef[cluster_num] * multivariate_normal.pdf(x, self.mu[cluster_num],
                                                                                self.sigma[cluster_num])

        return numerator / denominator

    # recompute all the parameters for all gaussian distributions.
    def m(self):
        # calculate the new means
        for k in range(self.mu.shape[0]):
            sum = 0.0
            for datum in self.data:
                sum += self.calc_responsibility(k, datum) * datum
            self.mu[k] = (self.mix_coef[k] * self.data.shape[0]) ** -1 * sum

        # calculate the new covariances
        for k in range(self.sigma.shape[0]):
            sum = 0.0
            for datum in self.data:
                mx = datum - self.mu[k]
                sum += self.calc_responsibility(k, datum) * np.outer(mx, mx)
            self.sigma[k] = (self.mix_coef[k, 0] *
                             self.data.shape[0]) ** -1 * sum

        # calculate new mixing coefficients
        for k in range(self.mix_coef.shape[0]):
            sum = 0.0
            for datum in self.data:
                sum += self.calc_responsibility(k, datum)
            self.mix_coef[k] = sum / self.data.shape[0]

    # assign each datapoint to the cluster that has the highest responsibility.
    def assign_data(self):
        self.assignments = np.zeros((self.data.shape[0], 1))
        for datum, index in zip(self.data, range(self.data.shape[0])):
            best_gauss = None
            for kth in range(self.k):  # find highest responsibility
                score = self.calc_responsibility(kth, datum)
                if best_gauss is None or score > best_gauss[1]:
                    best_gauss = kth, score
            self.assignments[index] = best_gauss[0]

    # given a certain epsilon value, return true if the difference between the previous log likelihood and the
    # new log likelihood is within the epsilon range. Otherwise return false.
    def check_convergence(self):
        new_log = 0.0
        for datum in self.data:  # compute the new log likelihood.
            inner_sum = 0.0
            for kth in range(self.mu.shape[0]):
                inner_sum += self.mix_coef[kth] * multivariate_normal.pdf(
                    datum, self.mu[kth], self.sigma[kth])
            if inner_sum == 0:
                new_log += -math.inf
                break
            else:
                new_log += math.log(inner_sum)
        # if difference is within the epsilon, return true.
        if self.prev_log is not None and abs(new_log - self.prev_log) < self.converge_epsilon:
            self.prev_log = new_log
            return True
        else:
            self.prev_log = new_log
            return False

    # train a model until convergence or until 100 iterations where performed.
    def train(self):
        iter = 0
        while not self.check_convergence():
            if iter == 100:
                break
            self.m()
            iter += 1

    # plot the data without colors.
    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.data[:, 0], self.data[:, 1], ".", color="lightblue")
        ax.plot(self.mu[:, 0], self.mu[:, 1], "x", color="black")
        ax.set_title('kmeans')
        plt.show()

    # plot and save data with colors that represent what cluster each belongs to.
    def plot_color(self, s):
        self.assign_data()
        cycol = cycle('bgrcmk')
        fig, ax = plt.subplots()
        ax.plot(0, 0, "x", color="black")
        ax.set_title('Gmm ' + s)
        for kth in range(self.mu.shape[0]):
            col = next(cycol)
            for datum in range(self.data.shape[0]):
                if self.assignments[datum] == kth:
                    ax.plot(self.data[datum, 0],
                            self.data[datum, 1], ".", c=col, alpha=.4)
        ax.plot(self.mu[:, 0], self.mu[:, 1], "o", color="black")
        plt.show()
        name = s + str(".pdf")
        fig.savefig(name, bbox_inches='tight')

    # run gmm em r times and return the object that had the highest log-likelihood.
    @staticmethod
    def run_r_iterations(r, k, data):
        best = None
        for i in range(r):
            new = Gmm(data)
            new.set_k(k)
            new.init_par()
            new.train()
            if best is None or best.prev_log < new.prev_log:
                best = new
        return best
