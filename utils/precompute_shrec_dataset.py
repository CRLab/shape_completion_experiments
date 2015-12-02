import h5py
string_dtype = h5py.special_dtype(vlen=bytes)
from multiprocessing import Pool
import os
from multiprocessing import Process, Queue
from datasets.shrec_reconstruction_dataset import ShrecReconstructionDataset, \
    build_training_example_scaled

PATCH_SIZE = 30


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
            scale_filepath = recon_dataset.examples[index][3]

            f = open(scale_filepath)
            line_0 = f.readline()
            offset_x, offset_y, offset_z, scale = line_0.split()
            custom_scale = float(scale)
            custom_offset = (float(offset_x), float(offset_y), float(offset_z))

            try:
                x, y = build_training_example_scaled(model_filepath,
                                                     pose_filepath,
                                                     single_view_pointcloud_filepath,
                                                     PATCH_SIZE,
                                                     custom_scale=custom_scale,
                                                     custom_offset=custom_offset)
                examples_queue.put((index, x, y, single_view_pointcloud_filepath, pose_filepath, model_filepath))
            except:
                examples_queue.put((index, None, None, single_view_pointcloud_filepath, pose_filepath, model_filepath))


#def main():


if __name__ == '__main__':
    #main()
    pc_dir = '/srv/data/shape_completion_data/shrec/gazebo_reconstruction_data_uniform_rotations_shrec_centered_scaled/'
    models_dir = '/srv/data/downloaded_mesh_models/shrec/models/'
    h5_base_dir = '/srv/data/shape_completion_data/shrec/h5/'

    model_names = os.listdir(pc_dir)

    for model_name in model_names:
        h5_dir = h5_base_dir + model_name + '/'
        if not os.path.exists(h5_dir):
            os.mkdir(h5_dir)

        if os.path.isfile(h5_dir + model_name + '.h5'):
            print('Skipping File: ' + h5_dir + model_name + '.h5')
            continue

        recon_dataset = ShrecReconstructionDataset(models_dir, pc_dir, model_name, patch_size=PATCH_SIZE)
        num_examples = recon_dataset.get_num_examples()


        print("Number of examples: " + str(num_examples))
        h5_dset = h5py.File(h5_dir + model_name + '.h5', 'w')

        h5_dset.create_dataset('x', (num_examples, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1), chunks=(100, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1))
        h5_dset.create_dataset('y', (num_examples, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1), chunks=(100, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1))


        h5_dset.create_dataset('single_view_pointcloud_filepath', (num_examples, 1), dtype=string_dtype)
        h5_dset.create_dataset('pose_filepath', (num_examples, 1), dtype=string_dtype)
        h5_dset.create_dataset('model_filepath', (num_examples, 1), dtype=string_dtype)

        h5_dset.close()
        index_queue = Queue()
        examples_queue = Queue(maxsize=100)

        print("staring readers")
        num_readers = 6

        for i in range(num_readers):
            reader_p = Process(target=reader, args=(index_queue, examples_queue))
            reader_p.daemon = True
            reader_p.start()

        print("putting indices on queue")
        for i in range(num_examples):
            index_queue.put(i)

        print("putting done statments on queue")
        for i in range(num_readers):
            index_queue.put('DONE')

        print("staring to write examples to h5dset")
        for i in range(num_examples):

            print("working on number: " + str(i))

            index, x, y, single_view_pointcloud_filepath, pose_filepath, model_filepath = examples_queue.get()
            h5_dset = h5py.File(h5_dir + model_name + '.h5')
            h5_dset['single_view_pointcloud_filepath'][index] = single_view_pointcloud_filepath
            h5_dset['pose_filepath'][index] = pose_filepath
            h5_dset['model_filepath'][index] = model_filepath

            if x is None or y is None:
                print("skipping index: " + str(index))
                h5_dset.close()
                continue

            h5_dset['x'][index] = x
            h5_dset['y'][index] = y
            h5_dset.close()