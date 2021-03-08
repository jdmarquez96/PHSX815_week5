import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
import math 
from scipy import integrate
import sympy as sym

def f(x): #gumbel density distribution
	return np.exp(-(x + np.exp(-x)))
x = sym.Symbol('x', real=True) #use to get the exact answer
ff = sym.exp(-(x + sym.exp(-x)))
size = np.arange(-3, 3, 0.01)


#trapezoid integration method
def trap(f, b, a, n): #f is function, and n is the number of rectangles
	h = (b-a)/float(n)
	intgr = 0.5 * h*(f(a) + f(b))
	for i in range(1, int(n)):
		intgr += f(a + i*h)
	intgr *= h
	return intgr

print("trapezoidal integration", trap(f, 3, -3, 100))


#gaussian quadrature
quad = integrate.quadrature(f,-3, 3)
print("Quadrature integration", quad)

#regualar integration
answer = sym.integrate(ff, (x, -3,3))
print("True answer",answer)
Vanswer = np.exp(-np.exp(-3)) - np.exp(-np.exp(3))
print("True answer value", Vanswer)


difference = trap(f, 3, -3, 100) - quad[0]
Adiff = Vanswer - trap(f, 3, -3, 100)
Adiff2 = Vanswer - quad[0]

print("Difference between two integration",difference)
print("Differene between true and trapezoid", Adiff)
print("Difference between true and quadratrue", Adiff2) 
print("Considering how small the difference is, I would say the difference is a good indicator of the actual error")



#plt.figure()
#plt.grid(True)
#plt.plot(size, f(size))
#plt.show()


