from __future__ import print_function, division
import cv2
from tqdm import tqdm
from torch import Tensor, cat

from image.normalization import normalize_image

from torchvision.utils import make_grid


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
            loss = loss_fn(theta, tnf_batch['theta_GT'].reshape([16, 6]))
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
                log_images(tb_writer,
                           batch, theta,
                           batch_idx, 'Model Output')
                log_images(tb_writer,
                           tnf_batch, tnf_batch['theta_GT'],
                           batch_idx, 'Ground Truth')

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
            loss = loss_fn(theta, tnf_batch['theta_GT'].reshape([16, 6]))
        else:
            loss = loss_fn(theta, tnf_batch['theta_GT'])

        val_loss += loss.data.cpu().numpy().item()

        # if possible log on TB an image_a, image_b couple with the relative predicted transform
        if coupled and tb_writer and batch_idx == 0:
            log_images(tb_writer, batch, theta, epoch)

    val_loss /= len(dataloader)
    print('Validation set: Average loss: {:.4f}'.format(val_loss))
    if tb_writer:
        tb_writer.add_scalar('val loss',
                             val_loss,
                             epoch)

    return val_loss


def log_images(tb_writer, batch, tnf_matrices, counter, n_max=1, tag=None):
    """
    Fn to log image batches

    :param tb_writer: Summary Writer
    :param batch: Batch of samples
    :param tnf_matrices: Batch of transformations to apply
    :param counter: Epoch index
    :param tag: Default None, if a string is specified tags the log
    with it as a prefix
    :return: None
    """

    try:
        images = zip(batch['image_a'], batch['image_b'], tnf_matrices)
    except KeyError:
        images = zip(batch['source_image'], batch['target_image'], tnf_matrices)

    for idx, (img_a, img_b, aff_matrix) in enumerate(images):
        if idx < n_max:

            denorm_img_a = normalize_image(img_a.unsqueeze(0),
                                           forward=False)
            denorm_img_b = normalize_image(img_b.unsqueeze(0),
                                           forward=False)
            transform = aff_matrix.cpu().detach().reshape([2, 3]).numpy()

            # convert to gray-scale the reshaped image in the correct order:
            # (height, width, n_channels)
            gray_img_a = cv2.cvtColor(denorm_img_a.squeeze(0).permute(1,
                                                                      2,
                                                                      0).cpu().numpy(),
                                      cv2.COLOR_BGR2GRAY)

            # we have to warp the normalized points and then denormalize the image
            # apply predicted affine transformation
            rows, cols = denorm_img_a.squeeze(0).shape[1:]
            gray_a_warp_on_b = cv2.warpAffine(gray_img_a,
                                              transform,
                                              (cols, rows))
            a_warp_on_b = Tensor(cv2.cvtColor(gray_a_warp_on_b,
                                              cv2.COLOR_GRAY2BGR)).permute(2,
                                                                           0,
                                                                           1).unsqueeze(0)

            couple_imgs = cat([denorm_img_a,
                               denorm_img_b])

            if not tag:
                log_name = 'sample_A/sample_B'
                warp_name = 'A warp on B'
            elif isinstance(tag, str):
                log_name = '{}\tsample_A/sample_B'.format(tag)
                warp_name = '{}\tA warp on B'.format(tag)
            else:
                raise ValueError("Unexpected type for 'tag', must be of type string.")

            tb_writer.add_images(log_name,
                                 make_grid(couple_imgs).unsqueeze(0),
                                 counter)
            tb_writer.add_images(warp_name,
                                 a_warp_on_b,
                                 counter)

        else:
            break

    return
