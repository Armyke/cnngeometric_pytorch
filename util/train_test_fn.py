from __future__ import print_function, division
import cv2
import numpy as np
from tqdm import tqdm
from torch import Tensor, cat

from image.normalization import normalize_image


def train(epoch, model, loss_fn, optimizer,
          dataloader, pair_generation_tnf,
          log_interval=50, tb_writer=None, scheduler=False):
    """
    Main function for training

    :param epoch: int, epoch index
    :param model: pytorch model object
    :param loss_fn: loss function of the model
    :param optimizer: optimizer of the model
    :param dataloader: DataLoader object
    :param pair_generation_tnf: Function to serve couples of samples
    :param log_interval: int, number of steps before logging scalars
    :param tb_writer: pytorch TensorBoard SummaryWriter
    :param scheduler: Eventual Learning rate scheduler

    :return: float, avg value of loss fn over epoch
    """

    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc='Epoch {}'.format(epoch))):
        optimizer.zero_grad()
        tnf_batch = pair_generation_tnf(batch)
        theta = model(tnf_batch)

        if loss_fn._get_name() == 'MSELoss':
            loss = loss_fn(theta, tnf_batch['theta_GT'].view([-1, 6]))

        elif loss_fn._get_name() == 'GridLossWithMSE':
            loss = loss_fn(theta, tnf_batch['theta_GT'],
                           tb_writer, (epoch - 1) * len(dataloader) + batch_idx)

        else:
            loss = loss_fn(theta, tnf_batch['theta_GT'])

        loss.backward()
        optimizer.step()

        # log loss
        tb_writer.add_scalar('training loss',
                             loss.data.item(),
                             (epoch - 1) * len(dataloader) + batch_idx)

        if scheduler:
            scheduler.step()
            if tb_writer:
                tb_writer.add_scalar('learning rate',
                                     scheduler.get_lr()[-1],
                                     (epoch - 1) * len(dataloader) + batch_idx)

        train_loss += loss.data.cpu().numpy().item()

        # log every log_interval
        if batch_idx % log_interval == 0:
            print('\tLoss: {:.6f}'.format(loss.data.item()))
            if tb_writer:
                log_images(tb_writer,
                           batch, theta,
                           (epoch - 1) * len(dataloader) + batch_idx,
                           'Model Output')

    train_loss /= len(dataloader)
    print('Train set: Average loss: {:.4f}'.format(train_loss))
    return train_loss


def validate_model(model, loss_fn,
                   dataloader, pair_generation_tnf,
                   epoch, tb_writer=None, coupled=False):
    """
    Sets the model to eval() mode and evaluates

    :param epoch: int, epoch index
    :param model: pytorch model object
    :param loss_fn: loss function of the model
    :param dataloader: DataLoader object
    :param pair_generation_tnf: Function to serve couples of samples
    :param epoch: epoch index
    :param tb_writer: pytorch TensorBoard SummaryWriter
    :param coupled: Bool, whether the dataset is coupled or not
    :return:
    """

    model.eval()
    val_loss = 0
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        tnf_batch = pair_generation_tnf(batch)
        theta = model(tnf_batch)

        if loss_fn._get_name() == 'MSELoss':
            loss = loss_fn(theta, tnf_batch['theta_GT'].view([-1, 6]))

        elif loss_fn._get_name() == 'GridLossWithMSE':
            loss = loss_fn(theta, tnf_batch['theta_GT'])

        else:
            loss = loss_fn(theta, tnf_batch['theta_GT'])

        val_loss += loss.data.cpu().numpy().item()

        # if possible log on TB an image_a, image_b couple with the relative predicted transform
        if coupled and tb_writer and batch_idx == 0:
            log_images(tb_writer, batch, theta, epoch,
                       tag='Validation',
                       n_max=len(batch['image_a']))

    val_loss /= len(dataloader)
    print('Validation set: Average loss: {:.4f}'.format(val_loss))
    if tb_writer:
        tb_writer.add_scalar('val loss',
                             val_loss,
                             epoch)

    return val_loss


def log_images(tb_writer, batch, tnf_matrices, counter, tag=None, n_max=1):
    """
    Fn to log image batches

    :param tb_writer: Summary Writer
    :param batch: Batch of samples
    :param tnf_matrices: Batch of transformations to apply
    :param counter: Epoch index
    :param tag: Default None, if a string is specified tags the log
     with it as a prefix
    :param n_max: Maximum numbers of images per batch to display
    :return: None
    """

    images = zip(batch['image_a'], batch['image_b'], batch['vertices_a'], tnf_matrices)

    for idx, (img_a, img_b, vertices, aff_matrix) in enumerate(images):
        if idx < n_max:

            denorm_a_img = normalize_image(img_a.unsqueeze(0),
                                           forward=False)
            denorm_b_img = normalize_image(img_b.unsqueeze(0),
                                           forward=False)
            transform = aff_matrix.cpu().detach().reshape([2, 3]).numpy()

            # get vertices to warp
            vertices = np.array(vertices)
            # add ones to warping points
            to_warp_pts = np.hstack([vertices,
                                     np.ones(shape=(len(vertices), 1))])

            # warp points through affine matrix
            warped_pts = transform.dot(to_warp_pts.T).T

            # denormalize warped points
            out_img_y, out_img_x = denorm_b_img.shape[2:]
            src_img_y, src_img_x = denorm_a_img.shape[2:]

            original_pts = np.array([[int(point[0] * src_img_x), int(point[1] * src_img_y)] for point in to_warp_pts],
                                    np.int32).reshape((-1, 1, 2))
            to_draw_pts = np.array([[int(point[0] * out_img_x), int(point[1] * out_img_y)] for point in warped_pts],
                                   np.int32).reshape((-1, 1, 2))

            drawn_a_img = np.moveaxis(denorm_a_img.squeeze().numpy(), 0, 2)
            drawn_b_img = np.moveaxis(denorm_b_img.squeeze().numpy(), 0, 2)

            drawn_a_img = np.ones(drawn_a_img.shape) * np.array(drawn_a_img * 255,
                                                                dtype=np.uint8)
            drawn_b_img = np.ones(drawn_b_img.shape) * np.array(drawn_b_img * 255,
                                                                dtype=np.uint8)

            # draw warped points over template image
            cv2.polylines(drawn_b_img, [to_draw_pts],
                          True, (0, 0, 255), 7)
            cv2.polylines(drawn_a_img, [original_pts],
                          True, (0, 0, 255), 7)

            # concatenate A and drawn B
            concat_img = cat([Tensor(drawn_a_img).double() / 255,
                              Tensor(drawn_b_img).double() / 255], 1)

            if not tag:
                log_name = 'A warp on B'
            elif isinstance(tag, str):
                log_name = '{}\tA warp on B'.format(tag)
            else:
                raise ValueError("Unexpected type for 'tag', must be of type string.")

            # log image
            tb_writer.add_images(log_name,
                                 concat_img.permute(2, 0, 1).unsqueeze(0),
                                 counter)

        else:
            break
