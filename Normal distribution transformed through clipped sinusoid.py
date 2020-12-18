###########################################################
# This program calculates a probability distribution of sampling points
# from a sine function, clipped at 0.7. This has direct applications 
# in communications sytems, where the clipping is caused by voltage saturation
# phenomena.
#
# The prorgram also compares ths distribution to the theoretical pdf obtained
# through a Jacobian transformation.
#
# Author: Mihai Ilas
#
###########################################################

import numpy as np
import matplotlib.pyplot as plt
import math

def ksdensity(data, width=0.3):
    """Returns kernel smoothing function from data points in data"""
    def ksd(x_axis):
        def n_pdf(x, mu=5., sigma=3.): # normal pdf
            u = (x - mu) / abs(sigma)
            y = (1 / (np.sqrt(2 * np.pi) * abs(sigma)))
            y *= np.exp(-u * u / 2)
            return y
        prob = [n_pdf(x_i, data, width) for x_i in x_axis]
        pdf = [np.average(pr) for pr in prob] # each row is one x value
        return np.array(pdf)
    return ksd

N=1000        #number of samples drawn

#clipped sinusoid function (at 0.7)
def f(x):
    y = []
    for i in range(len(x)):
       y.append(min(math.sin(x[i]),0.7))
    return y
        

fig, ax = plt.subplots(1)
x = np.random.uniform(0,2*math.pi,N)    #uniform distribution between 0 and 2*pi


y = f(x)        #map the uniform through the clipped sinusoid function
ax.hist(y, bins = 30)           #histogram obtained distribution

x = np.random.randn(N)     #standard gaussian

ks_density = ksdensity(x, width=0.3)  #create custom kernel density filter from x
x_values = np.linspace(-0.99, 0.99, 1000000)  #range of new distribution is (-1,1)
y_values=[]

#obtain and plot theoretical pdf through a Jacobian transformation 
for i in range(len(x_values)):
    if x_values[i] < 0.7:
        y_values.append(70 / (math.pi * math.sqrt(1 - (x_values[i] ** 2))))   #theoretical distribution
    elif 0.69 < x_values[i] < 0.71: y_values.append(1000 / (2 * math.pi * 0.7))
    else: y_values.append(0)

ax.plot(x_values, y_values,"red")  #print theoretical pdf





