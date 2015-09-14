
import numpy as np
import matplotlib.pyplot as plt
num_objects_list = [1, 5, 10, 25, 50]



plots = []
for num_objects in num_objects_list:
    label = 'Trained on ' + str(num_objects) + ' objects'

    if num_objects == 1:
        label = 'Trained on ' + str(num_objects) + ' object'
    jac_error = np.loadtxt('reconstruction_results_novel_view_shrec/' + str(num_objects) + '/3d_reconstruction_shrec_relu_jaccard.txt')
    jac_error = np.mean(jac_error.reshape(-1, 8), axis=1)
    X = np.array(range(jac_error.shape[0]))*20*32/1000
    Y = jac_error
    print X.shape
    print Y.shape
    plots.append(plt.plot(X, Y, label=label))


legend = plt.legend(loc=4,
           ncol=1, borderaxespad=0.)

for leg_obj in legend.legendHandles:
    leg_obj.set_linewidth(5.0)

plt.title("Novel View Jaccard Similarity")
plt.xlabel("Examples Seen (in Thousands)")
plt.ylabel("Jaccard Similarity")

plt.show()
