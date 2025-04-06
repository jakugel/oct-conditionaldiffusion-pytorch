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

import torch


def ddpm_schedules(beta1, beta2, T, type='linear'):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    if type == 'linear':
        beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    elif type == 'quadratic':
        beta_t = torch.linspace(beta1 ** 0.5, beta2 ** 0.5, T + 1) ** 2
    elif type == 'sigmoid':
        betas = torch.linspace(-6, 6, T + 1)
        beta_t = torch.sigmoid(betas) * (beta2 - beta1) + beta1
    else:
        print("Invalid type")
        return None

    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }
