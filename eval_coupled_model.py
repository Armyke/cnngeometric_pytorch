from __future__ import print_function, division
import os
from glob import glob
from argparse import ArgumentParser

import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader

from data.coupled_dataset import CoupledDataset
from data.download_datasets import download_PF_willow

from image.normalization import NormalizeImageDict, normalize_image

from util.torch_util import load_torch_model

from geotnf.transformation import CoupledPairTnf

"""

Script to evaluate a trained model as presented in the CNNGeometric CVPR'17 paper
on the ProposalFlow dataset

"""

print('CNNGeometric Coupled evaluation script')


def parse_flags():
    """
    Fn to parse arguments to pass to main func

    :return: args: Object which attributes are the flags values accessible through args.flag
    """

    # Argument parsing
    parser = ArgumentParser(description='CNNGeometric PyTorch implementation')
    # Paths
    parser.add_argument('--pretrained',
                        default='trained_models/best_pascal_checkpoint_adam_affine_grid_loss_resnet_random.pth.tar',
                        help='Trained affine model filename')
    parser.add_argument('--feature_extraction_cnn', type=str, default='vgg',
                        help='Feature extraction architecture: vgg/resnet101')
    parser.add_argument('--pf_path', type=str, default='datasets/PF-dataset',
                        help='Path to PF dataset csv')
    parser.add_argument('--pf_images_dir', type=str, default='datasets/PF-dataset',
                        help='Path to PF dataset images dir')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Directory where results will be saved')
    parser.add_argument('--trained_models_fn', type=str, default='checkpoint_adam',
                        help='trained model filename')
    parser.add_argument('--template_path', type=str, required=True,
                        help='Path to the template on which to test alignment')

    return parser.parse_args()


def test_and_save_alignment(batch, batch_index, model_output, out_dir):
    # load the template image and the batch to align
    img_a = batch['image_a'][0]
    img_b = batch['image_b'][0]

    denorm_a_img = np.array(normalize_image(img_a.unsqueeze(0),
                                            forward=False).squeeze(0).permute(1, 2, 0) * 255,
                            dtype=np.uint8)

    denorm_b_img = np.array(normalize_image(img_b.unsqueeze(0),
                                            forward=False).squeeze(0).permute(1, 2, 0) * 255,
                            dtype=np.uint8)

    # get vertices to warp
    vertices = batch['vertices_a'][0]
    # add ones to warping points
    to_warp_pts = np.hstack([vertices,
                             np.ones(shape=(len(vertices), 1))])

    aff_matrix = model_output[0].reshape(2, 3)

    transform = aff_matrix.detach().numpy()

    # warp points through affine matrix
    warped_pts = transform.dot(to_warp_pts.T).T

    # denormalize warped points
    out_img_y, out_img_x = denorm_b_img.shape[:2]
    src_img_y, src_img_x = denorm_a_img.shape[:2]

    original_pts = np.array([[int(point[0]*src_img_x), int(point[1]*src_img_y)] for point in to_warp_pts],
                            np.int32).reshape((-1, 1, 2))
    to_draw_pts = np.array([[int(point[0]*out_img_x), int(point[1]*out_img_y)] for point in warped_pts],
                           np.int32).reshape((-1, 1, 2))

    drawn_b_image = np.ones(denorm_b_img.shape) * denorm_b_img
    drawn_a_image = np.ones(denorm_a_img.shape) * denorm_a_img

    # draw warped points over template image
    cv2.polylines(drawn_b_image,  [to_draw_pts],
                  True, (0, 0, 255), 7)
    cv2.polylines(drawn_a_image,  [original_pts],
                  True, (0, 0, 255), 7)

    # concatenate A and drawn B and save image
    concat_img = np.concatenate([drawn_a_image, drawn_b_image], axis=1)

    out_path = os.path.join(out_dir, 'drawn_{}.png'.format(batch_index))

    cv2.imwrite(out_path, concat_img)

    return


def main(args):
    use_cuda = torch.cuda.is_available()

    # Download dataset if needed
    download_PF_willow('datasets/')

    # Create model
    print('Creating CNN model...')
    model_aff = load_torch_model(args, use_cuda)

    csv_path_list = glob(os.path.join(args.pf_path, '*.csv'))
    if len(csv_path_list) > 1:
        print("!!!!WARNING!!!! multiple csv files found, using first in glob order")
    elif not len(csv_path_list):
        raise FileNotFoundError("No csvs where found in the specified path!!!")

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    csv_path = csv_path_list[0]

    # Dataset and dataloader
    dataset = CoupledDataset(csv_file=csv_path,
                             training_image_path=args.pf_images_dir,
                             transform=NormalizeImageDict(['image_a', 'image_b']))  # template=args.template_path)

    batch_size = 1

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)

    # Set Tnf pair generation func
    pair_generation_tnf = CoupledPairTnf(use_cuda=use_cuda)

    for i, batch in enumerate(dataloader):

        tnf_batch = pair_generation_tnf(batch)
        theta = model_aff(tnf_batch)

        test_and_save_alignment(batch, i, theta, args.out_dir)


if __name__ == '__main__':
    ARGS = parse_flags()
    main(ARGS)
