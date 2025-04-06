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

import torch.nn as nn
import torch
from schedules import ddpm_schedules


class cDDPMMixed(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1, loss="MSE", scheduler_type="linear",
                 restrict_l_mode=None, restrict_u_mode=None, restrict_l_thresh=0, restrict_u_thresh=0):
        super(cDDPMMixed, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T, scheduler_type).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.restrict_l_mode = restrict_l_mode
        self.restrict_u_mode = restrict_u_mode
        self.restrict_l_thresh = restrict_l_thresh
        self.restrict_u_thresh = restrict_u_thresh

        if loss == "MSE":
            self.loss = nn.MSELoss()
        elif loss == "Huber":
            self.loss = nn.HuberLoss()
        elif loss == "MAE":
            self.loss = nn.L1Loss()

    def forward(self, x, c, unlabelled=False):
        """
        this method is used in training, so samples t and noise randomly
        """

        # potentially restrict the steps for which labelled and/or unlabelled training is performed
        if unlabelled:
            if self.restrict_u_mode is None:
                _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
            elif self.restrict_u_mode == 'first':
                _ts = torch.randint(1, self.restrict_u_thresh + 1, (x.shape[0],)).to(self.device)
            elif self.restrict_u_mode == 'last':
                _ts = torch.randint(self.restrict_u_thresh, self.n_T + 1, (x.shape[0],)).to(self.device)
        else:
            if self.restrict_l_mode is None:
                _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
            elif self.restrict_l_mode == 'first':
                _ts = torch.randint(1, self.restrict_l_thresh + 1, (x.shape[0],)).to(self.device)
            elif self.restrict_l_mode == 'last':
                _ts = torch.randint(self.restrict_l_thresh, self.n_T + 1, (x.shape[0],)).to(self.device)

        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
                self.sqrtab[_ts, None, None, None] * x
                + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability if this is a labelled batch
        # otherwise dropout context completely if it is an unlabelled batch
        if not unlabelled:
            context_mask = torch.bernoulli(torch.zeros_like(c) + self.drop_prob).to(self.device)
        else:
            context_mask = torch.bernoulli(torch.zeros_like(c) + 1.0).to(self.device)

        # return MSE between added noise, and our predicted noise
        return self.loss(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, guide_w=0.0, num_classes=10, sampling_drop_mode=None, sampling_drop_thresh=0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = torch.arange(0, num_classes).to(device)  # context for us just cycles throught the class labels
        c_i = c_i.repeat(int(n_sample / c_i.shape[0]))

        # don't drop context at test time (unless using new sampling technique)
        context_mask = torch.zeros_like(c_i).to(device)

        # for new sampling technique
        context_free_mask = torch.zeros_like(c_i).to(device)

        # double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1.  # makes second half of batch context free

        # construct context free mask of same shape
        context_free_mask = context_free_mask.repeat(2)
        context_free_mask[:] = 1.

        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}', end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            # double batch
            x_i = x_i.repeat(2, 1, 1, 1)
            t_is = t_is.repeat(2, 1, 1, 1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            if sampling_drop_mode is None:
                eps = self.nn_model(x_i, c_i, t_is, context_mask)
            elif sampling_drop_mode == 'first':
                if i < sampling_drop_thresh:
                    eps = self.nn_model(x_i, c_i, t_is, context_free_mask)
                else:
                    eps = self.nn_model(x_i, c_i, t_is, context_mask)
            elif sampling_drop_mode == 'last':
                if i > sampling_drop_thresh:
                    eps = self.nn_model(x_i, c_i, t_is, context_free_mask)
                else:
                    eps = self.nn_model(x_i, c_i, t_is, context_mask)

            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            x_i = x_i[:n_sample]
            x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
            )

        return x_i