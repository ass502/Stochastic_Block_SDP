import approx_three_communities
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
a_values = np.arange(0,.8,.08)
b_values = np.arange(1.6,6.6,.5)

#create matrix to store results of each experiment
#results = np.zeros((len(beta_values),len(alpha_values)))

for i,b in enumerate(b_values):
	for j,a in enumerate(a_values):
		
		print a,b
		start_time = time.time()

		#run iterations
		avg_entropy = approx_three_communities.run_iterations(n_iter,m,k,a,b)

		results[i,j] = avg_entropy
		
		print 'Minutes: ', (time.time()-start_time)/60.0

pickle.dump(results, open( "experiment_output/three_communities_"+str(m)+"_verts.p", "wb" ))

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

plt.savefig('experiment_output/three_communities_'+str(m)+'_verts.png')
