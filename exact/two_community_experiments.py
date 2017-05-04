import two_communities
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle

n = 100
n_iter = 50
upper_bound = int(math.floor(n / math.log(n)))
lower_bound = 0

"""n=20
n_iter = 10
upper_bound = 6
lower_bound = 0"""

results = np.zeros((upper_bound,upper_bound))
for beta in range(lower_bound,upper_bound):
	for alpha in range(beta+1,upper_bound+1):
		success_rate = two_communities.run_iterations(n_iter,n,alpha,beta+0.1)

		results[beta,alpha-1] = success_rate

pickle.dump(results, open( "experiment_output/2_communities_results.p", "wb" ))

imgplot = plt.imshow(results,cmap='gray')
axes = plt.gca()
axes.set_xticklabels(range(lower_bound,upper_bound+1),ha='center')
axes.set_ylabel('beta')
axes.set_xlabel('alpha')
axes.set_title('Two community results for '+str(n_iter)+' iterations')

plt.savefig('experiment_output/two_communities.png')
