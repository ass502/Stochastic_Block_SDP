import two_communities
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import sys

n = 250
n_iter = 20

#create range of values to try
#generally use small values, with alpha > beta
beta_values = np.arange(0,1.26,.06)
alpha_values = np.arange(1.5,5.7,.2)

#create matrix to store results of each experiment
results = np.zeros((len(beta_values),len(alpha_values)))

for i,beta in enumerate(beta_values):
	for j,alpha in enumerate(alpha_values):
		success_rate = two_communities.run_iterations(n_iter,n,alpha,beta)

		results[i,j] = success_rate

pickle.dump(results, open( "experiment_output/two_communities_250_verts_results.p", "wb" ))

imgplot = plt.imshow(results,cmap='gray')
axes = plt.gca()

axes.set_ylabel('beta')
axes.set_xlabel('alpha')
axes.set_title('Two community results for '+str(n)+' vertices')

#create labels for x and y axis
axes.yaxis.set_ticks(beta_values)
axes.xaxis.set_ticks(alpha_values)
axes.set_yticklabels(beta_values,ha='center')
axes.set_xticklabels(alpha_values,ha='center')

#draw boundary curve for sqrt(alpha) - sqrt(beta) = sqrt(k)
x_vals = np.arange(np.sqrt(2),5.7,0.1)
y_vals = np.power((np.sqrt(x_vals) - np.sqrt(2)),2)
axes.autoscale(False)
axes.plot(x_vals,y_vals,"r-")

plt.savefig('experiment_output/two_communities_250_verts.png')
