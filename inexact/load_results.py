import matplotlib.pyplot as plt
import math
import pickle
import sys
import numpy as np

n = 160
k=4

b_values = np.arange(0.1,.9,.08)
a_values = np.arange(3.7,6.7,.3)

results = pickle.load(open( "experiment_output/four_communities_40_verts.p", "rb" ))

imgplot = plt.imshow(results,cmap='gray', extent=[3.55,6.55,.86,.06], aspect="auto")
axes = plt.gca()

axes.set_ylabel('b')
axes.set_xlabel('a')
axes.set_title('Four community results for '+str(n)+' vertices')

axes.yaxis.set_ticks(b_values)
axes.xaxis.set_ticks(a_values)
axes.set_yticklabels(b_values)
axes.set_xticklabels(a_values,ha='center')

x_vals = np.arange(3.7,7,.3)
y_vals = (k**2 - k + 2*x_vals-k*np.sqrt(k**2-2*k+4*x_vals+1))/2
axes.autoscale(False)
axes.plot(x_vals,y_vals,"r-")

#plt.show()
plt.savefig('final_plots/four_communities_160_verts.png')