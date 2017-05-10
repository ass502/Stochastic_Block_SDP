import matplotlib.pyplot as plt
import math
import pickle
import sys
import numpy as np

n = 150
k=3

beta_values = np.arange(0,.8,.08)
alpha_values = np.arange(1.6,6.6,.5)

results = pickle.load(open( "experiment_output/three_communities_50_verts.p", "rb" ))

imgplot = plt.imshow(results,cmap='gray', extent=[1.35,6.35,.76,-.04], aspect="auto")
axes = plt.gca()

axes.set_ylabel('beta')
axes.set_xlabel('alpha')
axes.set_title('Three community results for '+str(n)+' vertices')

axes.yaxis.set_ticks(beta_values)
axes.xaxis.set_ticks(alpha_values)
axes.set_yticklabels(beta_values)
axes.set_xticklabels(alpha_values,ha='center')

x_vals = np.arange(3.1,7.1,.5)
y_vals = np.power(np.sqrt(x_vals) - np.sqrt(k),2)
axes.autoscale(False)
axes.plot(x_vals,y_vals,"r-")

#plt.show()
plt.savefig('final_plots/three_communities_150_verts.png')