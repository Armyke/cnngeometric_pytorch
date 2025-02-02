from __future__ import print_function, division

import os
import ast
from copy import deepcopy

import cv2
import pandas as pd
import numpy as np

from torch import Tensor
from torch.utils.data import Dataset
from torch.autograd import Variable

from geotnf.transformation import GeometricTnf


class CoupledDataset(Dataset):
    """

    Synthetically transformed pairs dataset for training with strong supervision

    Args:
            csv_file (string): Path to the csv file with image names and transformations.
            training_image_path (string): Directory with all the images.
            transform (callable): Transformation for post-processing the training pair
            (eg. image normalization)

    Returns:
            Dict: {
                   'image_a': dataset image cropped over vertices
                   'image_b': transformation destination image
                   'vertices_a': vertices of image a to warp
                   'theta': transformation from src (a) to dst (b)
                   }

    """

    def __init__(self, csv_file, training_image_path, output_size=(480, 640),
                 geometric_model='affine', transform=None):
        self.out_h, self.out_w = output_size
        # read csv file
        self.train_data = pd.read_csv(csv_file)
        self.img_a_names = self.train_data.iloc[:, 0]

        self.img_a_vertices = self.train_data.iloc[:, 2]
        self.img_b_names = self.train_data.iloc[:, 1]
        self.theta_array = self.train_data.iloc[:, 3:].values.astype('float')
        # copy arguments
        self.training_image_path = training_image_path
        self.transform = transform
        self.geometric_model = geometric_model
        self.affineTnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, use_cuda=False)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):

        # read image
        img_name_a = os.path.join(self.training_image_path, self.img_a_names[idx])

        img_name_b = os.path.join(self.training_image_path, self.img_b_names[idx])

        image_a = cv2.imread(img_name_a, cv2.IMREAD_COLOR)
        image_b = cv2.imread(img_name_b, cv2.IMREAD_COLOR)
        vertices = ast.literal_eval(self.img_a_vertices[idx])

        # read theta
        theta = self.theta_array[idx, :]

        if self.geometric_model == 'affine':

            # reshape theta to 2x3 matrix [A|t] where
            # first row corresponds to X and second to Y
            theta = theta[[3, 2, 5, 1, 0, 4]].reshape(2, 3)

        elif self.geometric_model == 'tps':

            theta = np.expand_dims(np.expand_dims(theta, 1), 2)

        # hold in the image_a only the crop but maintaining resolution
        # we achieve this by blanking each pixel outside the vertices
        image_a = blank_outside_verts(image_a, vertices)

        # make arrays float tensor for subsequent processing
        image_a = Tensor(image_a.astype(np.float32))
        image_b = Tensor(image_b.astype(np.float32))
        theta = Tensor(theta.astype(np.float32))

        # permute order of image to CHW
        image_a = image_a.transpose(1, 2).transpose(0, 1)
        image_b = image_b.transpose(1, 2).transpose(0, 1)

        # Resize image using bilinear sampling with identity affine tnf
        if image_a.size()[0] != self.out_h or image_a.size()[1] != self.out_w:

            image_a = self.affineTnf(Variable(image_a.unsqueeze(0), requires_grad=False)
                                     ).data.squeeze(0)

        # Resize image using bilinear sampling with identity affine tnf
        if image_b.size()[0] != self.out_h or image_b.size()[1] != self.out_w:

            image_b = self.affineTnf(Variable(image_b.unsqueeze(0), requires_grad=False)
                                     ).data.squeeze(0)

        # if self.mode == 'test':
        sample = {'image_a': image_a, 'vertices_a': Tensor(vertices),
                  'image_b': image_b, 'theta': theta}

        if self.transform:
            sample = self.transform(sample)

        return sample


def blank_outside_verts(src_image, element_vertices):
    """
    This method takes annotated vertices of an image and sets to 255
    the pixels outside vertices + margin

    :param src_image: np_array
    :param element_vertices: list of tuples of vertices
    :return: blanked_image: np_array
    """

    image = deepcopy(src_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # TODO add a little margin of pixels ?
    image = np.array(image, np.uint8)
    img_y, img_x = image.shape

    # denormalize vertices

    tmp_element_vertices = deepcopy(element_vertices)
    vertices = np.array([[int(x * img_x), int(y * img_y)] for x, y in tmp_element_vertices])

    # make mask leaving zero the outside of the vertices
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [vertices], -1, (255, 255, 255), -1, cv2.LINE_AA)

    # the filtered starting image is simply the and operator on each pixel
    # the background remains black though
    tmp_dst = cv2.bitwise_and(mask, image)

    # the tilde operator makes the bitwise not of the image
    # so summing, each part remained black outside the crop, goes white
    dst = tmp_dst + ~mask

    return cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
