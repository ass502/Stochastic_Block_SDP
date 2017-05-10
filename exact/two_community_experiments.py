import two_communities
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import sys
import time

n = 200
n_iter = 10

#create range of values to try
#generally use small values, with alpha > beta
beta_values = np.arange(0,1.2,.12)
alpha_values = np.arange(1.2,5.2,.4)

#create matrix to store results of each experiment
results = np.zeros((len(beta_values),len(alpha_values)))

start = time.time()

for i,beta in enumerate(beta_values):
	for j,alpha in enumerate(alpha_values):
		success_rate = two_communities.run_iterations(n_iter,n,alpha,beta)

		results[i,j] = success_rate
		current = time.time()
		print "beta,alpha: " +str(beta)+", "+str(alpha)
		print "time: " + str((current-start)/60)

pickle.dump(results, open( "experiment_output/two_communities_200_verts_results.p", "wb" ))

imgplot = plt.imshow(results,cmap='gray', extent=[1,5,1.14,-.06], aspect="auto")
axes = plt.gca()

axes.set_ylabel('beta')
axes.set_xlabel('alpha')
axes.set_title('Two community results for '+str(n)+' vertices')

axes.yaxis.set_ticks(beta_values)
axes.xaxis.set_ticks(alpha_values)
axes.set_yticklabels(beta_values)
axes.set_xticklabels(alpha_values,ha='center')

x_vals = np.arange(2,5.6,.4)
y_vals = np.power(np.sqrt(x_vals) - np.sqrt(2),2)
axes.autoscale(False)
axes.plot(x_vals,y_vals,"r-")

plt.savefig('experiment_output/two_community_200_verts.png')
