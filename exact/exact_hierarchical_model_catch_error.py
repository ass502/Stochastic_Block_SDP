import cvxopt
from cvxpy import *
import math
import numpy as np
import scipy.sparse as sparse

import sys

def create_matrix_B(m,k,alpha=5,beta=1):
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
	p = alpha * math.log(n) / n
	q = beta * math.log(n) / n

	#create true label of communities
	g = []
	for i in range(k):
		temp = [i]*m
		g.extend(temp)
	g = np.asarray(g)

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

def solve_sdp(n,B,g):

	#if a subproblem only consists of one true community, it is solved
	if len(set(g))==1:
		return True

	#initialize psd matrix
	X = Semidef(n)

	obj = Maximize(trace(B*X))

	constraints = [diag(X) == np.ones(n)]

	# Form and solve problem.
	prob = Problem(obj, constraints)
	prob.solve(solver='SCS')

	#find leading eigenvector of solved X
	try:
		w,v = sparse.linalg.eigs(X.value, k=1)
	except:
		return False

	#extract sign of each value in eigenvector
	estimated_labels = [x[0] for x in np.sign(v)]

	#check how the k true classes are distributed among the two estimated labels of 1 and -1
	comm_1_labels = set()
	comm_2_labels = set()
	comm_1_members = []
	comm_2_members = []
	for i in range(len(g)):
		if np.equal(estimated_labels[i],1):
			comm_1_labels.add(g[i])
			comm_1_members.append(i)
		else:
			comm_2_labels.add(g[i])
			comm_2_members.append(i)

	#exact recovery failed if a true label appears in both estimated communities, 
	#or if one estimated community is empty
	if comm_1_labels.intersection(comm_2_labels)!=set([]) or min(len(comm_1_labels),len(comm_2_labels))==0:
		return False

	#divide problem into subproblems by feeding community 1 and its corresponding portions of matrix B and vector g into SDP solver
	#and likewise for community 2
	return solve_sdp(len(comm_1_members),B[comm_1_members][:,comm_1_members],g[comm_1_members]) and \
		 solve_sdp(len(comm_2_members),B[comm_2_members][:,comm_2_members],g[comm_2_members]) 

def run_iterations(n_iter,m,k,alpha,beta):
	n = m*k

	success_count = 0
	for i in range(n_iter):
		B,g = create_matrix_B(m,k,alpha,beta)

		success_count += 1 if solve_sdp(n,B,g) else 0

	return success_count*1.0/n_iter

def main():
    success_rate = run_iterations(50,15,4,8,1)
    print success_rate

if __name__ == "__main__":
    main()