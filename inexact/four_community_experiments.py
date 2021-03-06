import approx_four_communities_binary
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import pickle
import sys

#n = 240
k = 4
m = 40
n_iter = 15

#create range of values to try
#generally use small values, with alpha > beta
b_values = np.arange(0.1,.9,.08)
a_values = np.arange(3.7,6.7,.3)

#create matrix to store results of each experiment
results = np.zeros((len(b_values),len(a_values)))

for i,b in enumerate(b_values):
	for j,a in enumerate(a_values):
		
		print a,b
		start_time = time.time()

		#run iterations
		avg_entropy = approx_four_communities_binary.run_iterations(n_iter,m,k,a,b)

		results[i,j] = avg_entropy
		
		print 'Minutes: ', (time.time()-start_time)/60.0

pickle.dump(results, open( "experiment_output/four_communities_"+str(m)+"_verts.p", "wb" ))

imgplot = plt.imshow(results,cmap='gray')
axes = plt.gca()

axes.set_ylabel('b')
axes.set_xlabel('a')
axes.set_title('Four community results for '+str(m)+' vertices')

#create labels for x and y axis
axes.yaxis.set_ticks(b_values)
axes.xaxis.set_ticks(a_values)
axes.set_yticklabels(b_values,ha='center')
axes.set_xticklabels(a_values,ha='center')

#draw boundary curve for sqrt(alpha) - sqrt(beta) = sqrt(k)
x_vals = np.arange(np.sqrt(2),5.7,0.1)
y_vals = np.power((np.sqrt(x_vals) - np.sqrt(2)),2)
axes.autoscale(False)
axes.plot(x_vals,y_vals,"r-")

plt.savefig('experiment_output/four_communities_'+str(m)+'_verts.png')
