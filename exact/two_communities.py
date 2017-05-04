import cvxopt
from cvxpy import *
import math
import numpy as np
import scipy.sparse as sparse

def create_matrix_B(n,alpha=9,beta=1):
	"""
	n = number of vertices in graph (number of members across all communities)

	p = probability of edge between two vertices in the same community
	q = probability of edge between two vertices in different communities
	alpha = constant parameter for draw p
	beta = constant parameter for draw q

	g = vector containing true labels of community membership
	A = adjacency matrix
	B = 2A - (1 x 1^T - I)
	"""

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

def run_iterations(n_iter,n,alpha,beta):
	success_count = 0
	for i in range(n_iter):
		B,g = create_matrix_B(n,alpha,beta)

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

		#print "estimated g:", estimated_labels
		#print "actual g:",g
		#print "Exact recovery achieved: " + str(np.array_equal(g,estimated_labels))

		success_count += 1 if np.array_equal(g,estimated_labels) else 0

	return success_count*1.0/n_iter

def main():
    success_rate = run_iterations(50,100,6,1)
    print success_rate

if __name__ == "__main__":
    main()
