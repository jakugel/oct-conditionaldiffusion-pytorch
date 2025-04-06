# oct-conditionaldiffusion-pytorch
Code for the paper "Diffusion model data augmentation for enhancing OCT retinal and choroidal segmentation" using conditional diffusion models (cDDPMs) synthesing OCT patches to provide enhanced data augmentation for patch-based segmentation methods. Leverages unlabelled data through joint training in addition to other new training techniques.

We use a conditional denoising diffusion probabilistic model (cDDPM), a type of diffusion model, to synthesise OCT image
patches of a number of classes, that can be added to the training dataset to enhance performance through data augmentation.

Paper link: currently under review

If the code and methods here are useful to you and aided in your research, please consider citing the papers above.

**Code**

We build upon the code at https://github.com/TeaPearce/Conditional_Diffusion_MNIST/ adding the following:
- code modularisation
- network flexibility
- support for several parameters including image size, diffusion steps, number of classes, batch size, number of network features,
    number of network layers, scheduler beta parameters, context dropout probability, loss function, scheduler type, network upsampling type
- in particular, we also expand the cDDPM training regime in a novel way to incorporate unlabelled data to further boost
    diversity of the synthesised images.
- we also incorporate a novel sampling process where different portions of the reverse diffusion process
    can be performed context free (instead of with context), controllable with parameters,
    allowing for potentially even greater diversity.
- similarly we incorporate another modification to the training regime that restricts the diffusion steps where labelled
    and/or unlabelled training are performed, again controllable with parameters.
