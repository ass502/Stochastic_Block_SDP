import cvxopt
from cvxpy import *
import math
import numpy as np
import scipy.sparse as sparse

def create_matrix_B(n,alpha=3,beta=1):
	#define draw probabilities for intercommunity and intracommunity edges
	p = alpha * math.log(n) / n
	q = beta * math.log(n) / n

	#create true label of communities
	community_1 = np.ones(n/2)
	community_2 = -1*np.ones(n/2)
	g = np.append(community_1,community_2)

	#adjacency matrix
	A = np.zeros([n,n])

	for r in range(n):
		for c in range(r+1,n):
			#in the same community if they have the same sign
			if g[r] * g[c] == 1:
				A[r,c] = np.random.binomial(1,p)
				A[c,r] = A[r,c]		
			else:
				A[r,c] = np.random.binomial(1,q)
				A[c,r] = A[r,c]

	B = 2*A - (np.ones([n,n]) - np.identity(n))
				
	return B,g
	
n = 10
B,g = create_matrix_B(n)

#initialize psd matrix
X = Semidef(n)

obj = Maximize(trace(B*X))

constraints = [diag(X) == np.ones(n)]

# Form and solve problem.
prob = Problem(obj, constraints)
prob.solve(solver='SCS')


w,v = sparse.linalg.eigs(X.value, k=1)
print "estimated g:", v
print "actual g:",g
