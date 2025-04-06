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

from cDDPM import cDDPM
from cDDPMMixed import cDDPMMixed
from networks import ContextUnet
import torch
import numpy as np
import os


def gen_data(save_dir, checkpoint_num, num_images, num_class_samples=10, image_size=32,
                n_T=500, device=None, n_classes=10, n_feat=128, num_layers=2, synthetic_data_dir = "./cddpm_images_generated",
                beta1=1e-4, beta2=0.02, drop_prob=0.1, loss="MSE", scheduler_type="linear", upsample_type='transpose',
             unlabelled=False, guide_w=0.0, restrict_l_mode=None, restrict_u_mode=None,
             restrict_l_thresh=0, restrict_u_thresh=0,
             sampling_drop_mode=None, sampling_drop_thresh=0):

    if not os.path.isdir(synthetic_data_dir):
        os.makedirs(synthetic_data_dir)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not unlabelled:
        ddpm = cDDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, num_layers=num_layers, n_classes=n_classes,
                                         upsample_type=upsample_type), betas=(beta1, beta2), n_T=n_T,
                    device=device, drop_prob=drop_prob, loss=loss, scheduler_type=scheduler_type)
    else:
        ddpm = cDDPMMixed(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, num_layers=num_layers, n_classes=n_classes,
                                          upsample_type=upsample_type), betas=(beta1, beta2), n_T=n_T,
                     device=device, drop_prob=drop_prob, loss=loss, scheduler_type=scheduler_type,
                          restrict_l_mode=restrict_l_mode, restrict_u_mode=restrict_u_mode,
                          restrict_l_thresh=restrict_l_thresh, restrict_u_thresh=restrict_u_thresh)

    ddpm.to(device)

    ddpm.load_state_dict(
        torch.load(save_dir + "/model_" + str(checkpoint_num) + ".pth"))

    ddpm.eval()

    all_imgs = np.zeros((num_images, 1, image_size, image_size))

    with torch.no_grad():
        for i in range(int(num_images / (num_class_samples * n_classes))):
            n_sample = num_class_samples * n_classes
            x_gen = ddpm.sample(n_sample, (1, image_size, image_size), device, guide_w=guide_w,
                                sampling_drop_mode=sampling_drop_mode, sampling_drop_thresh=sampling_drop_thresh)

            x_gen_numpy = x_gen.cpu().detach().numpy()

            all_imgs[i * (num_class_samples * n_classes): (i + 1) * (num_class_samples * n_classes)] = x_gen_numpy

    np.save(synthetic_data_dir + "/synimages_model_" + str(checkpoint_num) + ".npy", all_imgs)

