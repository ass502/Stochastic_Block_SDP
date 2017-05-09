import cvxopt
from cvxpy import *
import math
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

def entropy(community_labels):

	'''
	Convert list of community labels to list of probability distribution over labels.
	Then return entropy: - \sum_i p_i log_e p_i . 
	'''

	size = float(len(community_labels))
	probs = [community_labels.count(label)/size for label in set(community_labels)]
	return -sum([ p*np.log(p) for p in probs ])


def network_entropy(communities):

	'''
	Compute overall entropy of community divisions in network as:
	1/(num nodes) * [ size(community_1)*entropy(community_1) + ... + size(community_n)*entropy(community_n) ]
	'''

	weighted_entropies = 0
	network_size = 0

	for community in communities:
		
		size = len(community)
		weighted_entropies += size * entropy(community) #update network entropy
		network_size += size #update network size

	return weighted_entropies/float(network_size)


def create_matrix_B(m,k,a,b):
	
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
	p = a/float(n)
	q = b/float(n)

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

	#This doesn't apply here
	'''
	#if a subproblem only consists of one true community, it is solved
	if len(set(g))==1:
		return True
	'''

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

	#check how the k true classes are distributed among the two estimated labels of 1 and -1
	comm_1_members = []
	comm_2_members = []
	for i in range(len(g)):
		if np.equal(estimated_labels[i],1):
			comm_1_members.append(i)
		else:
			comm_2_members.append(i)

	return comm_1_members, comm_2_members


def test():

	m = 10 #number of vertices in each community
	k = 2 #number of communities
	n = m*k #number of network vertices
	a = 5
	b = 1

	B, g = create_matrix_B(m,k,a,b) #g is true community labels. B = 2A - (1-I) with A as random SBM graph

	comm1, comm2 = solve_sdp(n,B,g) #first binary split on network nodes

	labels1 = [ g[i] for i in comm1 ]
	labels2 = [ g[i] for i in comm2 ] 

	#Report entropy
	entropy = network_entropy([labels1,labels2])

	print 'Total entropy: ', entropy
	print 'Community 1: ', labels1
	print 'Community 2: ', labels2


def run_iterations(n_iters,m,k,a,b):

	'''store the average entropy over all iters'''
	n = m*k #network size

	entropy_sum = 0

	for it in range(n_iters):

		B, g = create_matrix_B(m,k,a,b) #g is true community labels. B = 2A - (1-I) with A as random SBM graph

		comm1, comm2 = solve_sdp(n,B,g) #first binary split on network nodes

		labels1 = [ g[i] for i in comm1 ] 
		labels2 = [ g[i] for i in comm2 ] 

		#Add current entropy
		entropy_sum += network_entropy([labels1,labels2])
	
	#Return average entropy over all iterations
	return entropy_sum/float(n_iters)

def main():

	test()

if __name__ == '__main__':
	main()
