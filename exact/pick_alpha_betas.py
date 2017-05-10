import numpy as np
import math

k = 3
m = 70
n = k*m

beta_values = np.arange(0,.8,.08)
alpha_values = np.arange(1.6,6.6,.5)

beta_root = [np.sqrt(b) for b in beta_values]
alpha_root = [np.sqrt(a) for a in alpha_values]

counter = 0.
for b in beta_root:
	for a in alpha_root:
		counter += (a-b>=np.sqrt(k))
		print a*math.log(n)/n, b*math.log(n)/n

tot=len(alpha_values)*len(beta_values)

print counter/tot
print len(alpha_values), len(beta_values), tot