import exact_hierarchical_model_catch_error as exact_hierarchical_model
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import pickle
import sys

k = 4
m = 40
n_iter = 15

#create range of values to try
#generally use small values, with alpha > beta
beta_values = np.arange(0,.8,.08)
alpha_values = np.arange(3.9,8.9,.5)

#create matrix to store results of each experiment
results = np.zeros((len(beta_values),len(alpha_values)))

for i,beta in enumerate(beta_values):
	for j,alpha in enumerate(alpha_values):
		print alpha,beta
		start_time=time.time()
		success_rate = exact_hierarchical_model.run_iterations(n_iter,m,k,alpha,beta)
		results[i,j] = success_rate
		print 'Minutes: ', (time.time()-start_time)/60.0
		print 'Success_rate: ', success_rate

pickle.dump(results, open( "experiment_output/four_communities_160_verts.p", "wb" ))

imgplot = plt.imshow(results,cmap='gray', extent=[3.65,8.65,.76,-.04], aspect="auto")
axes = plt.gca()

axes.set_ylabel('beta')
axes.set_xlabel('alpha')
axes.set_title('Four community results for '+str(n)+' vertices')

axes.yaxis.set_ticks(beta_values)
axes.xaxis.set_ticks(alpha_values)
axes.set_yticklabels(beta_values)
axes.set_xticklabels(alpha_values,ha='center')

x_vals = np.arange(4.4,9.4,.5)
y_vals = np.power(np.sqrt(x_vals) - np.sqrt(k),2)
axes.autoscale(False)
axes.plot(x_vals,y_vals,"r-")

plt.savefig('experiment_output/four_communities_160_verts.png')
