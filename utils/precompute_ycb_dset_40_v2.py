import os
from multiprocessing import Process, Queue
from precompute_worker import PrecomputeWorker


def reader(index_queue):
    print "In Reader"
    worker = PrecomputeWorker()
    print "constructed worker"
    while True:
        msg = index_queue.get()
        if msg == 'DONE':
            break
        else:
            task = msg
            binvox_file_path = task["binvox_file_path"]
            model_pose_filepath = task["model_pose_filepath"]
            single_view_pointcloud_filepath = task["single_view_pointcloud_filepath"]
            x_filepath = task["x_filepath"]
            y_filepath = task["y_filepath"]
            patch_size = task["patch_size"]
            if os.path.exists(x_filepath) and os.path.exists(y_filepath):
                print "file exists so skipping: " + x_filepath
                continue

            try:
                worker.build_training_example_scaled(binvox_file_path,
                                                     model_pose_filepath,
                                                     single_view_pointcloud_filepath,
                                                     x_filepath,
                                                     y_filepath,
                                                     patch_size)

                print "finished: " + single_view_pointcloud_filepath
            except:
              print "failed on: " + str(single_view_pointcloud_filepath)
    print "reader finishing"


def get_tasks():
    pc_dir = "/home/jvarley/8_18_11_40/"
    tasks = []
    for model in os.listdir(pc_dir):
        model_path = pc_dir + model + "/pointclouds/"
        binvox_filepath = "/srv/data/shape_completion_data/ycb/" + model +"/models/" + model + ".binvox"
        for mfile in os.listdir(model_path):
            if "_pc.pcd" in mfile:

                single_view_pointcloud_filepath = model_path + mfile
                model_pose_filepath = model_path + mfile.replace("pc.pcd", "model_pose.npy")
                x_filepath = model_path + mfile.replace("pc.pcd", "x.pcd")
                y_filepath = model_path + mfile.replace("pc.pcd", "y.pcd")
                task = {}
                task["binvox_file_path"] = binvox_filepath
                task["model_pose_filepath"] = model_pose_filepath
                task["single_view_pointcloud_filepath"] = single_view_pointcloud_filepath
                task["x_filepath"] = x_filepath
                task["y_filepath"] = y_filepath
                task["patch_size"] = 40
                tasks.append(task)

    return tasks


if __name__ == "__main__":

    tasks = get_tasks()

    index_queue = Queue()

    print("starting readers")
    num_workers = 7

    readers = []
    for i in range(num_workers):
        reader_p = Process(target=reader,
                           args=(index_queue,))
        reader_p.daemon = True
        reader_p.start()
        readers.append(reader_p)

    print("putting indices on queue")
    for task in tasks:
        index_queue.put(task)

    print("putting done statements on queue")
    for i in range(num_workers):
        index_queue.put('DONE')

    for reader_p in readers:
        reader_p.join()

