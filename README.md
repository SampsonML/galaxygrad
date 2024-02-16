# galaxygrad
Package for diffusion model trained on HSC galaxies
See https://pypi.org/project/galaxygrad/0.1.2/
Contains 4 generative diffusion models ScoreNet32 and ScoreNet64 for both the HSC and ZTF surveys. These are used to return the gradients of an arbitrary image with respect to a prior distribution of individual artifact free galaxy models. Current functions include ScoreNetXX(image) returns gradients as stated. Data transformatons are now done inside the package.
