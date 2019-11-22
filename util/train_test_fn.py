from __future__ import print_function, division
import cv2
import numpy as np
from tqdm import tqdm
from torch import Tensor

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
            loss = loss_fn(theta, np.reshape(tnf_batch['theta_GT'], [16, 6]))
        else:
            loss = loss_fn(theta, tnf_batch['theta_GT'])

        loss.backward()
        optimizer.step()

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
                tb_writer.add_scalar('training loss',
                                     loss.data.item(),
                                     (epoch - 1) * len(dataloader) + batch_idx)

    train_loss /= len(dataloader)
    print('Train set: Average loss: {:.4f}'.format(train_loss))
    return train_loss


def validate_model(model, loss_fn,
                   dataloader, pair_generation_tnf,
                   epoch, tb_writer=None, coupled=False):

    model.eval()
    val_loss = 0
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        tnf_batch = pair_generation_tnf(batch)
        theta = model(tnf_batch)

        if loss_fn._get_name() == 'MSELoss':
            loss = loss_fn(theta, np.reshape(tnf_batch['theta_GT'], [16, 6]))
        else:
            loss = loss_fn(theta, tnf_batch['theta_GT'])

        val_loss += loss.data.cpu().numpy().item()

        # if possible log on TB an image_a, image_b couple with the relative predicted transform
        if coupled and tb_writer:
            log_images(tb_writer, batch, epoch)

    val_loss /= len(dataloader)
    print('Validation set: Average loss: {:.4f}'.format(val_loss))
    if tb_writer:
        tb_writer.add_scalar('val loss',
                             val_loss,
                             epoch)

    return val_loss


def log_images(tb_writer, batch, epoch):
    """
    Fn to log image batches

    :param tb_writer: Summary Writer
    :param batch: Batch of samples
    :param epoch: Epoch index
    :return: None
    """

    # load the image
    img_a = batch['image_a'][0]
    img_b = batch['image_b'][0]

    denorm_img_a = normalize_image(img_a.unsqueeze(0),
                                   forward=False)
    denorm_img_b = normalize_image(img_b.unsqueeze(0),
                                   forward=False)
    transform = batch['theta'][0].numpy()

    # TODO use already prepared methods to warp in geotnf.point_tnf
    # convert to gray-scale the reshaped image in the correct order:
    # (height, width, n_channels)
    gray_img_a = cv2.cvtColor(np.array((denorm_img_a * 255).squeeze(0).permute(1, 2, 0),
                                       dtype=np.uint8),
                              cv2.COLOR_BGR2GRAY)

    # TODO matrix is normalized so in order to have the correct warp:
    # we have to warp the normalized points and then denormalize the image
    # apply predicted affine transformation
    rows, cols = denorm_img_a.squeeze(0).shape[1:]
    gray_a_warp_on_b = cv2.warpAffine(gray_img_a,
                                      transform,
                                      (cols, rows))
    a_warp_on_b = Tensor(cv2.cvtColor(gray_a_warp_on_b,
                                      cv2.COLOR_GRAY2BGR)).permute(2, 0, 1)

    # log the three images with the tb_writer
    tb_writer.add_images('image A', denorm_img_a,
                         epoch)

    tb_writer.add_images('image B', denorm_img_b,
                         epoch)

    tb_writer.add_images('A warped on B', a_warp_on_b.unsqueeze(0) / 255,
                         epoch)

    return
