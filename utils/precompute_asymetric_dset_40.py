import os
from multiprocessing import Pool
from multiprocessing import Process, Queue
import h5py
#from datasets.ycb_reconstruction_dataset import YcbDataset, \
#    build_training_example_scaled

from datasets.asymetric_reconstruction_dataset import AsymetricDataset,\
    build_training_example_scaled

string_dtype = h5py.special_dtype(vlen=bytes)

PATCH_SIZE = 40


def reader(index_queue, examples_queue):
    while True:
        msg = index_queue.get()
        if msg == 'DONE':
            break
        else:
            index = msg
            single_view_pointcloud_filepath = recon_dataset.examples[index][0]
            pose_filepath = recon_dataset.examples[index][1]
            model_filepath = recon_dataset.examples[index][2]
            try:
                x, y = build_training_example_scaled(
                    model_filepath,
                    pose_filepath,
                    single_view_pointcloud_filepath,
                    PATCH_SIZE)
                examples_queue.put((
                    index, x, y, single_view_pointcloud_filepath,
                    pose_filepath, model_filepath))
            except:
              examples_queue.put((index, None, None,
                                  single_view_pointcloud_filepath,
                                  pose_filepath, model_filepath))


if __name__ == "__main__":
    
    data_dir = '/srv/data/shape_completion_data/asymetric/'
    for model_name in os.listdir(data_dir):
        
        models_dir = data_dir + model_name + '/'
        models_dir = '/srv/data/processed_mesh_models/asymetric/' + model_name
        pc_dir = data_dir + model_name + '/pointclouds/'
        h5_dir = data_dir + model_name + '/h5_40_v0/'
        if not os.path.exists(h5_dir):
            os.mkdir(h5_dir)

        recon_dataset = AsymetricDataset(data_dir, model_name, patch_size=PATCH_SIZE)
        num_examples = recon_dataset.get_num_examples()

        print("Number of examples: " + str(num_examples))
        h5_dset = h5py.File(h5_dir + model_name + '.h5', 'w')

        h5_dset.create_dataset(
            'x', (num_examples, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1),
            chunks=(100, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1))
        h5_dset.create_dataset(
            'y', (num_examples, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1),
            chunks=(100, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1))

        h5_dset.create_dataset('single_view_pointcloud_filepath',
                               (num_examples, 1), dtype=string_dtype)
        h5_dset.create_dataset('pose_filepath', (num_examples, 1),
                               dtype=string_dtype)
        h5_dset.create_dataset('model_filepath', (num_examples, 1),
                               dtype=string_dtype)

        h5_dset.close()
        index_queue = Queue()
        examples_queue = Queue(maxsize=100)

        print("starting readers")
        num_readers = 4

        for i in range(num_readers):
            reader_p = Process(target=reader,
                               args=(index_queue, examples_queue))
            reader_p.daemon = True
            reader_p.start()

        print("putting indices on queue")
        for i in range(num_examples):
            index_queue.put(i)

        print("putting done statements on queue")
        for i in range(num_readers):
            index_queue.put('DONE')

        print("starting to write examples to h5dset")
        for i in range(num_examples):

            print("working on number: " + str(i))

            (index, x, y, single_view_pointcloud_filepath, pose_filepath,
             model_filepath) = examples_queue.get()
            h5_dset = h5py.File(h5_dir + model_name + '.h5')
            h5_dset['single_view_pointcloud_filepath'][
                index] = single_view_pointcloud_filepath
            h5_dset['pose_filepath'][index] = pose_filepath
            h5_dset['model_filepath'][index] = model_filepath

            if x is None or y is None:
                print("skipping index: " + str(index))
                h5_dset.close()
                continue

            h5_dset['x'][index] = x
            h5_dset['y'][index] = y
            h5_dset.close()


