import csv
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    gt_to_partials = []
    gt_to_completions = []
    completed_to_closests = []
    with open('y16_m08_d17_h16_m22/jaccard.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # if "blue_wood_block_1inx1in" in row["single_view_pointcloud_filepath"]:
            #     print "skipping"
            #     continue

            gt_to_partial= float(row["gt_to_partial_jaccard"])
            gt_to_completion = float(row["gt_to_completion_jaccard"])
            completed_to_closest = float(row["completion_to_closest_training_example_jaccard"])

            if gt_to_completion < 0.2:
                continue

            if gt_to_completion < 0.4:
                print row
                continue

            gt_to_partials.append(gt_to_partial)
            gt_to_completions.append(gt_to_completion)
            completed_to_closests.append(completed_to_closest)

    gt_to_partial = np.array(gt_to_partials)
    gt_to_completion = np.array(gt_to_completions)
    completed_to_closest = np.array(completed_to_closests)
    #
    plt.scatter(list(completed_to_closest), list(gt_to_completion))
    plt.show()
    #
    # plt.scatter(gt_to_partial, gt_to_completion)
    # plt.show()

    import IPython
    IPython.embed()


