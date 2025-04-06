'''
Code for the paper "Diffusion model data augmentation for enhancing OCT retinal and choroidal segmentation"

We use a conditional denoising diffusion probabilistic model (cDDPM), a type of diffusion model, to synthesise OCT image
patches of a number of classes, that can be added to the training dataset to enhance performance through data augmentation.

We build upon the code at https://github.com/TeaPearce/Conditional_Diffusion_MNIST/ adding the following:
- code modularisation
- network flexibility
- support for several parameters including image size, diffusion steps, number of classes, batch size, number of network features,
    number of network layers, scheduler beta parameters, context dropout probability, loss function, scheduler type, network upsampling type
- in particular, we also expand the cDDPM training regime in a novel way to incorporate unlabelled data to further boost
    diversity of the synthesised images.
- we also incorporate a novel sampling process where different portions of the reverse diffusion process
    can be performed context free (instead of with context), controllable with parameters,
    allowing for potentially even greater diversity
- similarly we incorporate another modification to the training regime that restricts the diffusion steps where labelled
    and/or unlabelled training are performed, again controllable with parameters
'''

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image, make_grid
import numpy as np
import h5py
from cDDPM import cDDPM
from networks import ContextUnet
import os


def train_diffusion_model(h5_filename, images=None, img_labels=None, image_size=32, n_epoch=200, batch_size=128,
                n_T=500, device=None, n_classes=10, n_feat=128, num_layers=2, lrate=1e-4, save_model=True, save_model_dir = "./cddpm_models",
                          save_images=True, save_images_dir = "./cddpm_images",
                beta1=1e-4, beta2=0.02, drop_prob=0.1, loss="MSE", ema_multiplier=0.95, scheduler_type="linear", upsample_type='transpose',
                          sampling_drop_mode=None, sampling_drop_thresh=0):

    if save_model:
        if not os.path.isdir(save_model_dir):
            os.makedirs(save_model_dir)

    if save_images:
        if not os.path.isdir(save_images_dir):
            os.makedirs(save_images_dir)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)

    ddpm = cDDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, num_layers=num_layers, n_classes=n_classes, image_size=image_size,
                                      upsample_type=upsample_type),
                 betas=(beta1, beta2), n_T=n_T, device=device, drop_prob=drop_prob, loss=loss, scheduler_type=scheduler_type)
    ddpm.to(device)

    # load from the h5_file if it is provided
    if h5_filename is not None:
        h5_file = h5py.File(h5_filename, "r")

        images = h5_file["images"][0, :]
        images = np.reshape(images, newshape=(-1, image_size, image_size, 1))
        images = np.transpose(images, axes=(0, 3, 2, 1)) / 255

        img_labels = h5_file["labels"][0, :]
        img_labels = np.reshape(img_labels, newshape=(-1,))
    elif images is None or img_labels is None:
        print("No valid data provided in h5 file or as arguments. Exiting...")
        return

    tensor_x = torch.Tensor(images)
    tensor_y = torch.Tensor(img_labels)
    tensor_y = tensor_y.to(torch.int64)

    dset = TensorDataset(tensor_x, tensor_y)
    dloader = DataLoader(dset, batch_size=batch_size, shuffle=True)

    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)

        pbar = tqdm(dloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = ema_multiplier * loss_ema + (1 - ema_multiplier) * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        ddpm.eval()
        with torch.no_grad():
            n_sample = 4 * n_classes
            x_gen = ddpm.sample(n_sample, (1, image_size, image_size), device, sampling_drop_mode=sampling_drop_mode,
                                sampling_drop_thresh=sampling_drop_thresh)

            # append some real images at bottom, order by class also
            x_real = torch.Tensor(x_gen.shape).to(device)
            for k in range(n_classes):
                for j in range(int(n_sample / n_classes)):
                    try:
                        idx = torch.squeeze((c == k).nonzero())[j]
                    except:
                        idx = 0
                    x_real[k + (j * n_classes)] = x[idx]

            x_all = torch.cat([x_gen, x_real])
            grid = make_grid(x_all, nrow=10)
            save_image(grid, save_images_dir + f"/image_ep{ep}.png")
            print('saved image at ' + save_images_dir + f"/image_ep{ep}.png")

        # optionally save model
        if save_model:
            torch.save(ddpm.state_dict(), save_model_dir + f"/model_{ep}.pth")
            print('saved model at ' + save_model_dir + f"/model_{ep}.pth")
