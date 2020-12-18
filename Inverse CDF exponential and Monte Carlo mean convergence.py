###########################################################
# This program illustrates how to generate an exponential distribution
# from a more "common" uniform distribution, by using the inverse CDF method:
# if f(x) (in this case the exponential pdf) is continuous and derivable, by integrating
# to obtain the CDF and applying its inverse to x is a simple transformation to obtain the exponential
# distribution.
#
# The program also helps proving that in this case the Monte Carlo mean calculation
# method is not biased, by examining the convergence of the MSE between empirical and theoretical means
# as the size of samples drawn increases.
#
# Author: Mihai Ilas
#
###########################################################

import numpy as np
import matplotlib.pyplot as plt

#inverse of the CDF of the standard exponential distribution
def F_inverse(x):
    y = []
    for i in range(len(x)):
       y.append(-np.log(1 - x[i]))
    return y
        
N = 10000       #number of samples drawn 
fig, ax = plt.subplots(1)
#get uniform distribution samples
x = np.random.rand(N)    #uniform distribution samples


y=F_inverse(x)           #apply CDF inverse transformation
ax.hist(y, bins = 30)    #histogram the distributiom

#calculate mean and variance
mean=0
for i in range(len(y)):
    mean += y[i]
mean /= N
variance = 0
for i in range(len(y)):
    variance += ((y[i]-mean) ** 2) / N

print(mean)
print(variance)

N_vector = []
MSE_vector = []          #array of MSE errors between empirical mean and theoretical mean
step = 50
Range = 200
measurements_no=10
theoretical_mean=1

for i in range(1,Range):   #iterate over different sample sizes
    N_vector.append(i * step)
    average_mean = 0
    
    for j in range(measurements_no):    #multiple measurements of empirical mean for each sample size for more accuracy
        x = np.random.rand(i*step)
        y = F_inverse(x)
        mean = 0
        
        for k in range(len(y)):
            mean += y[k]
            
        mean /= i * step
        average_mean += mean  
        
    average_mean /= measurements_no       #average results over 10 measurements
    MSE_vector.append((average_mean - theoretical_mean) ** 2)
    
fig, ax = plt.subplots(1)
ax.plot(N_vector, MSE_vector, "red")    #plot MSE list to visualise convergence as N increases     

