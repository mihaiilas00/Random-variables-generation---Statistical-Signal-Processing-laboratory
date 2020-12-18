###################################################################
# The program generates a Variance-Gamma distribution (with slower decaying tails compared to a Gaussian),
# through the next steps:
#  •	Pick a value for parameter θ > 0. 
#  •	Generate a sample from the Gamma distribution, p(v) = G(v|θ, 1/θ). (a particular case with b=1/a of the general Gamma distribution)
#  •	Calculate the variance as u = 1/v. 
#  •	Sample from a Normal distribution p(x|u) = N (x|0, u). 
#  •	x is the random variable of interest.
# It also looks at the ranges of variables drawn to prove the tail differences.
#
# Author: Mihai Ilas
# 
###################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
def ksdensity(data, width=0.3):
    """Returns kernel smoothing function from data points in data"""
    def ksd(x_axis):
        def n_pdf(x, mu=0., sigma=1.): # normal pdf
            u = (x - mu) / abs(sigma)
            y = (1 / (np.sqrt(2 * np.pi) * abs(sigma)))
            y *= np.exp(-u * u / 2)
            return y
        prob = [n_pdf(x_i, data, width) for x_i in x_axis]
        pdf = [np.average(pr) for pr in prob] # each row is one x value
        return np.array(pdf)
    return ksd

N=100000    #number of samples drawn

fig, ax = plt.subplots(1)
x=[]

theta = 4    #shape of gamma distribution

v=np.random.gamma(theta,1 / theta,N)   #particular gamma distribuion with scale=1/shape

ax.hist(v, bins = 500,color = "green",density = True)   #histogram the gamma distributed samples
plt.show()

fig, ax = plt.subplots(1)
x = np.random.randn(N)    #draw N samples from a standard gaussian distribution

y=[]
for i in range(N):
     y.append(x[i] * (math.sqrt(abs(1 / v[i]))))   #dot product between the gaussian array and the gamma array essentially varying the variances of the distributions randomly according to v
     
#comparing the ranges of the gaussian and variance-gamma distribution     
range_gaussian = max(x)-min(x)
range_variance_gamma = max(y)-min(y)
print(range_gaussian)
print(range_variance_gamma)


ax.hist(x, bins = 500,color = "red",density = True) # number of bins

ax.hist(y, bins = 500, alpha = 0.5,density = True) # number of bins

plt.show()

fig, ax = plt.subplots(1)
ks_density = ksdensity(y, width = 0.5)
x_values = np.linspace(-25, 25, 500)
ax.plot(x_values, np.log(ks_density(x_values)),"red")  #plot logarithm of the kernel density estimate

plt.show()