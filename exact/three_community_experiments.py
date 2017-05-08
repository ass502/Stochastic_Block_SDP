import exact_hierarchical_model
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import pickle
import sys

#n = 240
k = 3
m = 50
n_iter = 15

#create range of values to try
#generally use small values, with alpha > beta
beta_values = np.arange(0,.8,.08)
alpha_values = np.arange(1.6,6.6,.5)

#create matrix to store results of each experiment
results = np.zeros((len(beta_values),len(alpha_values)))

for i,beta in enumerate(beta_values):
	for j,alpha in enumerate(alpha_values):
		print alpha,beta
		start_time=time.time()
		success_rate = exact_hierarchical_model.run_iterations(n_iter,m,k,alpha,beta)
		results[i,j] = success_rate
		print 'Minutes: ', (time.time()-start_time)/60.0

pickle.dump(results, open( "experiment_output/three_communities_50_verts.p", "wb" ))

imgplot = plt.imshow(results,cmap='gray')
axes = plt.gca()

axes.set_ylabel('beta')
axes.set_xlabel('alpha')
axes.set_title('Three community results for '+str(m)+' vertices')

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

plt.savefig('experiment_output/three_communities_50_verts.png')
