import numpy as np
import sys

np.random.seed(11)

# generate data
alpha_0 = 0.1
mu_0 = 5
K = 12
N = 1000

mu = np.random.gamma(alpha_0, mu_0 / alpha_0, K)
x = np.random.normal(mu, 1, (N,K))

# fit model
import numpy as np
import sys
from scipy.special import gammaln, digamma
np.random.seed(11)

## helper functions
# soft-plus
def SP(x):
    return np.log(1. + np.exp(x))

# derivative of soft-plus
def dSP(x):
    return np.exp(x) / (1. + np.exp(x))

# inverse of soft-plus
def iSP(x):
    return np.log(np.exp(x) - 1.)

# log probability of a gamma given sparsity a and mean m
def Gamma(x, a, m):
    return a * np.log(a) - a * np.log(m) - gammaln(a) + (a-1.) * np.log(x) - a * x / m

# derivative of above log gamma with respect to sparsity a
def dGamma_alpha(x, a, m):
    return np.log(a) + 1. - np.log(m) - digamma(a) + np.log(x) - (x / m)

# derivative of above log gamma with respect to mean m
def dGamma_mu(x, a, m):
    return - (a / m) + ((a * x) / m**2)

# log probabilty of a Normal distribution with unit variance and mean m
def Normal(x, m):
    return - 0.5 * (x - m)**2 - np.log(np.sqrt(2 * np.pi))

# covariance where each column is draws for a variable
# (returns a row of covariances)
def cov(a, b):
    v = (a - sum(a)/a.shape[0]) * (b - sum(b)/b.shape[0])
    return sum(v)/v.shape[0]

# variance with same structure as covariance above
def var(a):
    return cov(a,a)

## set up for inference
# the number of samples to draw for each parameter of iterest
S = 1024

# initialize
lambda_alpha = 0.1 * np.ones(K) # alpha_0 + 0.01 * np.exp(np.random.normal(0, 1, K))
# randomness in initialization is not needed here, but may be for other models
lambda_mu = 0.01 * np.ones(K) # np.exp(np.random.normal(0, 1, K)))

iteration = 0

# used for RMSprop
MS_mu = np.zeros(K)
MS_alpha = np.zeros(K)

# set up log file
logf = open("log.tsv", 'w+')
logf.write("iteration\tcomponent\tmean\tsparsity\n")

# log the truth used to simulate the data
for k in range(K):
    logf.write("-1\t%d\t%f\t%f\n" % (k, mu[k], 0))

# log the initial conditions
for k in range(K):
    logf.write("%d\t%d\t%f\t%f\n" % (iteration, k, SP(lambda_mu)[k], SP(lambda_alpha)[k]))

# run BBVI for a fixed number of iterations
while (iteration < 2000):
    sample_mus = np.random.gamma(SP(lambda_alpha), SP(lambda_mu) / SP(lambda_alpha), (S,K))

    # truncate samples (don't sample zero)
    sample_mus[sample_mus < 1e-300] = 1e-300

    # probability of samples given prior
    p = Gamma(sample_mus, alpha_0, mu_0)

    # probability of samples given variational parameters
    q = Gamma(sample_mus, SP(lambda_alpha), SP(lambda_mu))

    # gradients of variational parameters sparsity (alpha) and mean (mu) given samples
    g_alpha = dSP(lambda_alpha) * dGamma_alpha(sample_mus, SP(lambda_alpha), SP(lambda_mu))
    g_mu = dSP(lambda_mu) * dGamma_mu(sample_mus, SP(lambda_alpha), SP(lambda_mu))

    # probability of observations given samples
    for i in xrange(N):
        p += Normal(x[i], sample_mus)

    # control variates to decrease variance of gradient; one for each variational parameter
    cv_alpha = cov(g_alpha * (p - q), g_alpha) / var(g_alpha)
    cv_mu = cov(g_mu * (p - q), g_mu) / var(g_mu)

    # RMSprop: keep running average of gradient magnitudes
    # (the gradient will be divided by sqrt of this later)
    if MS_mu.all() == 0:
        MS_mu = (g_mu**2).sum(0)
        MS_alpha = (g_alpha**2).sum(0)
    else:
        MS_mu = 0.9 * MS_mu + 0.1 * (g_mu**2).sum(0)
        MS_alpha = 0.9 * MS_alpha + 0.1 * (g_alpha**2).sum(0)

    # Robbins-Monro sequence for step size
    print iteration
    rho = (iteration + 1024) ** -0.7

    # update each variational parameter with average over samples
    lambda_mu += rho * (1. / S) * (g_mu / np.sqrt(MS_mu) * (p - q - cv_mu)).sum(0)
    lambda_alpha += rho * (1. / S) * (g_alpha / np.sqrt(MS_alpha) * (p - q - cv_alpha)).sum(0)

    # truncate variational parameters
    lambda_alpha[lambda_alpha < iSP(0.005)] = iSP(0.005)
    #lambda_alpha[lambda_alpha < 0.005] = 0.005
    lambda_alpha[lambda_alpha > iSP(np.log(sys.float_info.max))] = iSP(np.log(sys.float_info.max))
    lambda_mu[lambda_mu > iSP(np.log(sys.float_info.max))] = iSP(np.log(sys.float_info.max))
    lambda_mu[lambda_mu < iSP(1e-5)] = iSP(1e-5)

    iteration += 1

    # log this iteration
    for k in range(K):
        logf.write("%d\t%d\t%f\t%f\n" % (iteration, k, SP(lambda_mu)[k], SP(lambda_alpha)[k]))
