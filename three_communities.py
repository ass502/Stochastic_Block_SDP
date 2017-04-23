import cvxopt
from cvxpy import *
import math
import numpy as np
import scipy.sparse as sparse

import sys

def create_matrix_B(m,k,alpha=8,beta=.5):
	"""
	m = number of vertices in each community
	k = number communities

	p = probability of edge between two vertices in the same community
	q = probability of edge between two vertices in different communities
	alpha = constant parameter for draw p
	beta = constant parameter for draw q

	g = vector containing true labels of community membership
	A = adjacency matrix
	B = 2A - (1 x 1^T - I)
	"""

	n = m*k

	#define draw probabilities for intercommunity and intracommunity edges
	p = alpha * math.log(m) / m
	q = beta * math.log(m) / m

	#create true label of communities
	g = []
	for i in range(k):
		temp = [i]*m
		g.extend(temp)

	#adjacency matrix
	A = np.zeros([n,n])

	for r in range(n):
		for c in range(r+1,n):
			#in the same community if they have the same value
			if g[r] == g[c]:
				A[r,c] = np.random.binomial(1,p)
				A[c,r] = A[r,c]		
			else:
				A[r,c] = np.random.binomial(1,q)
				A[c,r] = A[r,c]

	B = 2*A - (np.ones([n,n]) - np.identity(n))
				
	return B,g

m = 30
k = 3
n = m*k
B,g = create_matrix_B(m,k)

#initialize psd matrix
X = Semidef(n)

obj = Maximize(trace(B*X))

constraints = [diag(X) == np.ones(n)]

# Form and solve problem.
prob = Problem(obj, constraints)
prob.solve(solver='SCS')

#find leading eigenvector of solved X
w,v = sparse.linalg.eigs(X.value, k=1)

#extract sign of each value in eigenvector
estimated_labels = [x[0] for x in np.sign(v)]

#check how the 3 true classes are distributed among the two estimated labels of 1 and -1
comm_1 = set()
comm_2 = set()
for i in range(len(g)):
	if np.equal(estimated_labels[i],1):
		comm_1.add(g[i])
	else:
		comm_2.add(g[i])
print comm_1,comm_2

#exact recovery is achieved if one community consists of two of the true labels and the other community consists of only the third true label
print "Exact recovery achieved: " + str(comm_1.intersection(comm_2)==set([]) and max(len(comm_1),len(comm_2))==2 and min(len(comm_1),len(comm_2))==1)
