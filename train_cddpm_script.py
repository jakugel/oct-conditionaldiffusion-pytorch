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

from train_cddpm import train_diffusion_model
import torch

H5_FILENAME = "./data/your_data.hdf5"
IMAGES = None
IMG_LABELS = None
IMG_SIZE = 32
N_EPOCH = 200
BATCH_SIZE = 128
DIFFUSION_STEPS = 500
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_CLASSES = 10
N_FEAT = 128
NUM_LAYERS = 2
LEARNING_RATE = 1e-4
SAVE_MODEL = True
SAVE_MODEL_DIR = "./cddpm_models"
SAVE_IMAGES = True
SAVE_IMAGES_DIR = "./cddpm_images"
VAR_BETA1 = 1e-4
VAR_BETA2 = 0.02
DROP_PROB = 0.1
LOSS = "MSE"
EMA_MULTIPLIER = 0.95
SCHEDULER_TYPE = "linear"
UPSAMPLE_TYPE = "transpose"
SAMPLING_DROP_MODE = None
SAMPLING_DROP_THRESH = 0

train_diffusion_model(H5_FILENAME, IMAGES, IMG_LABELS, IMG_SIZE, N_EPOCH, BATCH_SIZE, DIFFUSION_STEPS, DEVICE, N_CLASSES,
                      N_FEAT, NUM_LAYERS, LEARNING_RATE, SAVE_MODEL, SAVE_MODEL_DIR, SAVE_IMAGES, SAVE_IMAGES_DIR,
                      VAR_BETA1, VAR_BETA2, DROP_PROB, LOSS, EMA_MULTIPLIER, SCHEDULER_TYPE, UPSAMPLE_TYPE,
                      SAMPLING_DROP_MODE, SAMPLING_DROP_THRESH)