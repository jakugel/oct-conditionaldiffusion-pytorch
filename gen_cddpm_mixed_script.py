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

from synthesise_patches import gen_data
import torch

IMG_SIZE = 32
DIFFUSION_STEPS = 500
NUM_IMAGES = 100
NUM_CLASS_SAMPLES = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_CLASSES = 10
N_FEAT = 128
NUM_LAYERS = 2
SAVE_MODEL_DIR = "./cddpm_mixed_models"
MODEL_EPOCH = 1
SAVE_IMAGES_DIR = "./cddpm_mixed_images_generated"
VAR_BETA1 = 1e-4
VAR_BETA2 = 0.02
DROP_PROB = 0.1
LOSS = "MSE"
EMA_MULTIPLIER = 0.95
SCHEDULER_TYPE = "linear"
UPSAMPLE_TYPE = "transpose"
RESTRICT_LABELLED_MODE = None
RESTRICT_UNLABELLED_MODE = None
RESTRICT_LABELLED_THRESH = 0
RESTRICT_UNLABELLED_THRESH = 0
SAMPLING_DROP_MODE = None
SAMPLING_DROP_THRESH = 0
UNLABELLED = True
GUIDANCE_W = 0.0

gen_data(SAVE_MODEL_DIR, MODEL_EPOCH, NUM_IMAGES, NUM_CLASS_SAMPLES, IMG_SIZE, DIFFUSION_STEPS, DEVICE, N_CLASSES,
                      N_FEAT, NUM_LAYERS, SAVE_IMAGES_DIR,
                      VAR_BETA1, VAR_BETA2, DROP_PROB, LOSS, SCHEDULER_TYPE, UPSAMPLE_TYPE, UNLABELLED, GUIDANCE_W,
                            RESTRICT_LABELLED_MODE, RESTRICT_UNLABELLED_MODE, RESTRICT_LABELLED_THRESH,
                            RESTRICT_UNLABELLED_THRESH, SAMPLING_DROP_MODE, SAMPLING_DROP_THRESH)