import numpy as np
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r
from Random import Random 


random= Random()


xmin = -2
xmax = 2
sig = 1
x = np.linspace(xmin, xmax, 10000)
mu = 0


def p(x):  #gumbel distribution/target distribution
	return np.exp(-(x + np.exp(-x)))

def q(x): #sine wave/ proposal distribution
	return   0.37*np.sin(x/2 + 1.5 )    #(1/np.sqrt(2*np.pi*sig**2))*np.exp(-(x-mu)**2/2*sig**2)


def SampleFlat():
        return xmin + (xmax-xmin)*random.rand()


data=[]
N = 1000000
for i in range(0, N):
	X = SampleFlat()
	R = p(X)/q(X)
	rand = random.rand()
	if (rand > R):
		continue
	else:
		data.append(X)
		i += 1



weights = np.ones_like(data) /len(data)


#def rejection(iter = 10000):
#	sample = []
#
#	for i in range(iter):
#		X = xmin + (xmax - xmin)*random.rand()
#		R = p(X)/q(X)
#		if (random.rand() > R):
#			continue
#		else:
#			sample.append(X)
#	return np.array(sample)

#s = rejection(iter = 10000)


plt.figure()
plt.grid(True)
plt.plot(x, p(x),label="Target")
plt.plot(x, q(x),label="Proposal")
plt.hist(data, 1000, density=True)
plt.legend()
plt.xlabel("data points")
plt.ylabel("count")
plt.title("Rejection Sampling")
plt.savefig("rejectionsampling.png")
plt.show()








