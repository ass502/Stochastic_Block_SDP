import os
import argparse
import cvxopt
from cvxpy import *
import math
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import pickle
import time

parser = argparse.ArgumentParser()
parser.add_argument('--niter', type=int, default=10, help='Number of experiments.')
parser.add_argument('--vert_per_comm', type=int, default=30, help='Number of vertices per community.')
parser.add_argument('--num_comm', type=int, default=4, help='Number of communities.')
parser.add_argument('--outdir', type=str, default='./', help='Directory to write output files to.')
parser.add_argument('--plotfile', type=str, default='entropies.jpg', help='Image path for histogram of entropies')
parser.add_argument('--entropies_pickle', type=str, default='entropies.p', help='Network entropy for each experiment')
parser.add_argument('--comms_pickle', type=str, default='', help='Comms for each experiments')
args = parser.parse_args()

try:
    os.makedirs(args.outdir)
except OSError:
    pass

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


def create_second_B_matrix(B, g, comm):

	'''Create a version of B and g which only includes the vertices in community comm'''

	return B[comm][:,comm], g[comm]


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


def test_binary():

	m = 10 #number of vertices in each community
	k = 4 #number of communities
	n = m*k #number of network vertices
	a = 1
	b = 2

	B, g = create_matrix_B(m,k,a,b) #g is true community labels. B = 2A - (1-I) with A as random SBM graph

	temp_left_comm, temp_right_comm = solve_sdp(n,B,g) #first binary split on network nodes

	#Now create another B matrix and g vector, corresponding to "left" side of binary tree
	B_left, g_left = create_second_B_matrix(B, g, temp_left_comm)	

	#And apply SDP to get first two communities (from left side of split tree)
	comm1, comm2 = solve_sdp(len(temp_left_comm), B_left, g_left) 

	#Now create another B matrix and g vector, corresponding to "left" side of binary tree
	B_right, g_right = create_second_B_matrix(B, g, temp_right_comm)	

	#And apply SDP to get first two communities (from left side of split tree)
	comm3, comm4 = solve_sdp(len(temp_right_comm), B_right, g_right) 

	labels1 = [ g_left[i] for i in comm1 ] #comm1 and comm2 indexes correspond to true labels in g_left
	labels2 = [ g_left[i] for i in comm2 ] 
	labels3 = [ g_right[i] for i in comm3 ] #comm3 and comm4 indexes correspond to true labels in g_right
	labels4 = [ g_right[i] for i in comm4 ]

	#Report entropy
	entropy = network_entropy([labels1,labels2,labels3,labels4])

	print 'Total entropy: ', entropy
	print 'Community 1: ', labels1
	print 'Community 2: ', labels2
	print 'Community 3: ', labels3
	print 'Community 4: ', labels4


def simulate(n_iters,m,k,a,b):

	'''
	m is number of vertices per community
	k is number of communities
	'''

	n = m*k #network size

	entropies = []
	if args.comms_pickle:
		comms = []

	for it in range(n_iters):

		if it%100==0:
			print 'iter ', it, '/', n_iters

		B, g = create_matrix_B(m,k,a,b) #g is true community labels. B = 2A - (1-I) with A as random SBM graph

		temp_left_comm, temp_right_comm = solve_sdp(n,B,g) #first binary split on network nodes

		#Now create another B matrix and g vector, corresponding to "left" side of binary tree
		B_left, g_left = create_second_B_matrix(B, g, temp_left_comm)	

		#And apply SDP to get first two communities (from left side of split tree)
		comm1, comm2 = solve_sdp(len(temp_left_comm), B_left, g_left) 

		#Now create another B matrix and g vector, corresponding to "left" side of binary tree
		B_right, g_right = create_second_B_matrix(B, g, temp_right_comm)	

		#And apply SDP to get first two communities (from left side of split tree)
		comm3, comm4 = solve_sdp(len(temp_right_comm), B_right, g_right) 

		labels1 = [ g_left[i] for i in comm1 ] #comm1 and comm2 indexes correspond to true labels in g_left
		labels2 = [ g_left[i] for i in comm2 ] 
		labels3 = [ g_right[i] for i in comm3 ] #comm3 and comm4 indexes correspond to true labels in g_right
		labels4 = [ g_right[i] for i in comm4 ]

		#Compute and store entropy
		entropies.append( network_entropy([labels1,labels2,labels3,labels4]) )

		#Store labels, maybe
		if args.comms_pickle:
			comms.append([labels1, labels2, labels3, labels4])

	#Pickle and plot results
	with open(args.outdir+args.entropies_pickle, 'wb') as out:
		pickle.dump(entropies, out)

	#Pickle communities, maybe
	if args.comms_pickle:
			with open(args.outdir+args.comms_pickle, 'wb') as out:
				pickle.dump(comms, out)

	#Histogram of recoveries
	plt.hist(entropies)
	plt.savefig(args.outdir+args.plotfile)


def run_iterations(n_iters,m,k,a,b):

	'''store the average entropy over all iters'''
	n = m*k #network size

	entropy_sum = 0

	for it in range(n_iters):

		#if it%100==0:
		#	print 'iter ', it, '/', n_iters

		B, g = create_matrix_B(m,k,a,b) #g is true community labels. B = 2A - (1-I) with A as random SBM graph

		temp_left_comm, temp_right_comm = solve_sdp(n,B,g) #first binary split on network nodes

		#Now create another B matrix and g vector, corresponding to "left" side of binary tree
		B_left, g_left = create_second_B_matrix(B, g, temp_left_comm)	

		#And apply SDP to get first two communities (from left side of split tree)
		comm1, comm2 = solve_sdp(len(temp_left_comm), B_left, g_left) 

		#Now create another B matrix and g vector, corresponding to "left" side of binary tree
		B_right, g_right = create_second_B_matrix(B, g, temp_right_comm)	

		#And apply SDP to get first two communities (from left side of split tree)
		comm3, comm4 = solve_sdp(len(temp_right_comm), B_right, g_right) 

		labels1 = [ g_left[i] for i in comm1 ] #comm1 and comm2 indexes correspond to true labels in g_left
		labels2 = [ g_left[i] for i in comm2 ] 
		labels3 = [ g_right[i] for i in comm3 ] #comm3 and comm4 indexes correspond to true labels in g_right
		labels4 = [ g_right[i] for i in comm4 ]

		#Add current entropy
		entropy_sum += network_entropy([labels1,labels2,labels3,labels4])
	
	#Return average entropy over all iterations
	return entropy_sum/float(n_iters)



def main():

	test_binary()

	'''
	start_time = time.time()
	simulate(args.niter,args.vert_per_comm,args.num_comm)

	with open(args.outdir+'params.txt', 'wb') as params_file:
		params_file.write('niter='+str(args.niter)+'\n')
		params_file.write('m='+str(args.vert_per_comm)+'\n')
		params_file.write('k='+str(args.num_comm)+'\n')

	print 'Time in minutes: ', (time.time()-start_time)/60.
	'''

if __name__=='__main__':
	main()

