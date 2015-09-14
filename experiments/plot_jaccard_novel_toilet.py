
import numpy as np
import matplotlib.pyplot as plt





jac_error = np.loadtxt('reconstruction_results_novel_view_shrec/' + str(num_objects) + '/3d_reconstruction_shrec_relu_jaccard.txt')
jac_error = np.mean(jac_error.reshape(-1, 8), axis=1)
X = np.array(range(jac_error.shape[0]))*20*32/1000
Y = jac_error
print X.shape
print Y.shape
plt.plot(X, Y)


legend = plt.legend(loc=4,
           ncol=1, borderaxespad=0.)

for leg_obj in legend.legendHandles:
    leg_obj.set_linewidth(5.0)

plt.title("Novel View Jaccard Similarity")
plt.xlabel("Examples Seen (in Thousands)")
plt.ylabel("Jaccard Similarity")

plt.show()
