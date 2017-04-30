'''
Note: can't do this by iteratively applying SBM to "non-pure" (wrt community label) communities bc that 
requires knowing community labels in between the hierarchical applications - this unavailable in practice.

Instead, we know for these experiments that there are three equal-sized communities. 
If you apply SBM once, you have two communities. Then you should apply SBM to the larger of the two, 
to have three communities. To measure the quality of the splits, we can measure the network entropy
according


'''

import cvxopt
from cvxpy import *
import math
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import pickle
import time

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
	p = alpha * math.log(m) / m
	q = beta * math.log(m) / m

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

def create_second_B_matrix(B_orig, g_orig, comm):

	'''Create a version of B and g which only includes the vertices in community comm'''

	g = [ g_orig[vertex] for vertex in comm ]

	len_comm = len(comm)
	B = np.zeros([len_comm,len_comm])

	for row in range(len_comm):
		for col in range(row+1,len_comm):

			#row and col refer to the new indexing for new B and g
			#but to compute B[row,col], look at comm[row] and comm[col] which is index value in B_orig and g_orig and 
			orig_row_idx = comm[row]
			orig_col_idx = comm[col]
			B[row,col] = B_orig[orig_row_idx,orig_col_idx]
			B[col,row] = B[row,col]

	return B, g


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


def test_hierarchy():

	m = 15 #number of vertices in each community
	k = 3 #number of communities
	n = m*k #number of network vertices

	B1, g1 = create_matrix_B(m,k) #g1 is true community labels. B1 = 2A - (1-I) with A as random SBM graph

	temp_comm1, temp_comm2 = solve_sdp(n,B1,g1) 

	#Figure out the bigger community to split again. If equal size, pick randomly.
	if len(temp_comm1) <= len(temp_comm2): 
		comm_1 = temp_comm1
		temp_comm = temp_comm2
	else:
		comm_1 = temp_comm2
		temp_comm = temp_comm1

	#Now create another B matrix and g vector
	B2, g2 = create_second_B_matrix(B1, g1, temp_comm)	

	#And apply SBM again
	comm_2, comm_3 = solve_sdp(len(g2),B2,g2)

	#Compute actual labels of communities
	labels_1 = [ g1[i] for i in comm_1 ] #comm_1 indexes correspond to the true labels in g1
	labels_2 = [ g2[i] for i in comm_2 ] #comm_2 and comm_3 indexes correspond to the true labels in g2
	labels_3 = [ g2[i] for i in comm_3 ]

	#Report entropy
	entropy = network_entropy([labels_1, labels_2, labels_3])
	
	print 'Total Entropy: ', entropy
	print 'Community 1: ', labels_1
	print 'Community 2: ', labels_2
	print 'Community 3: ', labels_3


def simulate(n_iters, outpickle, outplot):

	start_time = time.time()

	m = 15 #number of vertices in each community
	k = 3 #number of communities
	n = m*k #number of network vertices

	entropies = []
	for it in range(n_iters):

		if it%100==0:
			print 'iter ', it, '/', n_iters

		B1, g1 = create_matrix_B(m,k) #g1 is true community labels. B1 = 2A - (1-I) with A as random SBM graph

		temp_comm1, temp_comm2 = solve_sdp(n,B1,g1) 

		#Figure out the bigger community to split again. If equal size, pick randomly.
		if len(temp_comm1) <= len(temp_comm2): 
			comm_1 = temp_comm1
			temp_comm = temp_comm2
		else:
			comm_1 = temp_comm2
			temp_comm = temp_comm1

		#Now create another B matrix and g vector
		B2, g2 = create_second_B_matrix(B1, g1, temp_comm)	

		#And apply SBM again
		comm_2, comm_3 = solve_sdp(len(g2),B2,g2)

		#Compute actual labels of communities
		labels_1 = [ g1[i] for i in comm_1 ] #comm_1 indexes correspond to the true labels in g1
		labels_2 = [ g2[i] for i in comm_2 ] #comm_2 and comm_3 indexes correspond to the true labels in g2
		labels_3 = [ g2[i] for i in comm_3 ]

		#Compute and store entropy
		entropies.append( network_entropy([labels_1, labels_2, labels_3]) )

	#Pickle and plot results
	with open(outpickle, 'wb') as out:
		pickle.dump(np.sort(entropies), out)

	plt.hist(entropies)
	plt.savefig(outplot)

	print 'Time in minutes: ', (time.time()-start_time)/60.

def main():

	#test_hierarchy()
	simulate(100000, '100k.p', 'results_100k.jpg')

if __name__ == '__main__':
	main()

