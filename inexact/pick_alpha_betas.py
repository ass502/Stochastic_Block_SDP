import numpy as np
import math

def test_bound(a,b,k):

	if k==2:
		return (a-b)**2 > 2*(a+b) #KASTEN-STEIGEN BOUND
	else:
		return (a-b)**2/float(k*(a+(k-1)*b)) > 1

k = 4
m = 40
n = float(k*m)

b_values = np.arange(0,.8,.08)
a_values = np.arange(3.3,6.3,.3)

counter = 0.
for b in b_values:
	for a in a_values:
		counter += test_bound(a,b,k)
		print a/n, b/n
		
tot=len(a_values)*len(b_values)

print counter/tot
print len(a_values), len(b_values), tot