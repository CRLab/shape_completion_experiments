
import numpy as np


class OffHandler():

    def read(self, filepath):

        f = open(filepath, 'r')

        f.readline()

        v_count, f_count, e_count = [int(x) for x in f.readline().split()]

        self.vertices = np.zeros((v_count, 3))
        self.faces = np.zeros((f_count, 3))

        self.v_count = v_count
        self.f_count = f_count
        self.e_count = e_count

        i = 0
        while i < v_count:
            line = f.readline()
            self.vertices[i] = [float(x) for x in line.split()]
            i += 1

        i = 0
        while i < f_count:
            line = f.readline()
            num_indices, vid_0, vid_1, vid_2 = [float(x) for x in line.split()]
            self.faces[i] = [vid_0, vid_1, vid_2]
            i += 1

    def write(self, outfile):

        f = open(outfile, 'w')
        fs = ""
        fs += "OFF\n"
        fs += str(self.v_count) + " " + str(self.f_count) + " "  + str(self.e_count) + "\n"

        i = 0
        while i < self.v_count:
            fs += str(self.vertices[i])[1:-1] + "\n"
            i += 1

        i = 0
        while i < self.f_count:
            fs += "3 " + str(self.faces[i])[1:-1] + "\n"
            i += 1

        f.write(fs)

    def scale_mesh(self, scale_factor):
        self.vertices *= scale_factor

    def recenter_mesh(self):

        bbox = self.get_bounding_box()
        offset = [(bbox[1] - bbox[0])/2.0 + bbox[0],
                  (bbox[3] - bbox[2])/2.0 + bbox[2],
                  (bbox[5] - bbox[4])/2.0 + bbox[4]]

        self.vertices -= offset

        return offset

    def scale_and_center(self, desired_largest_side=.25):
        scale_factor = self.get_scale_factor(desired_largest_side)
        self.scale_mesh(scale_factor)
        offset = self.recenter_mesh()

        return offset, scale_factor

    def get_centroid(self):
        return np.average(self.vertices[:,0]), np.average(self.vertices[:,1]), np.average(self.vertices[:, 2])

    def get_full_vertices(self):
        return self.vertices

    def get_bounding_box(self):
        return self.vertices[:, 0].min(),\
               self.vertices[:, 0].max(),\
               self.vertices[:, 1].min(),\
               self.vertices[:, 1].max(),\
               self.vertices[:, 2].min(),\
               self.vertices[:, 2].max()

    def get_scale_factor(self, desired_largest_side=.25):
        bounding_box = self.get_bounding_box()

        ranges = [bounding_box[1] - bounding_box[0],
                  bounding_box[3] - bounding_box[2],
                  bounding_box[5] - bounding_box[4]
                  ]

        ranges.sort()

        largest_dim = ranges[-1]

        return desired_largest_side / largest_dim



