[![arXiv](https://img.shields.io/badge/arXiv-2401.07313-<COLOR>.svg)](https://arxiv.org/abs/2401.07313)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![PyPI](https://img.shields.io/pypi/v/galaxygrad?label=pypi%20package)
![PyPI - Downloads](https://img.shields.io/pypi/dm/galaxygrad)
# GalaxyGrad
Package for a score-based diffusion model trained on HSC galaxies, and ZTF like simulations.
See https://pypi.org/project/galaxygrad/0.1.5/
Contains 4 generative diffusion models {HSC, ZTF}_ScoreNet32 and {HSC, ZTF}ScoreNet64 for both the HSC and ZTF surveys. These are used to return the gradients of an arbitrary image with respect to a prior distribution of individual artifact free galaxy models. Current functions include {HSC, ZTF}_ScoreNet{32, 64}(image) returns gradients w,r,t the trained prior of the image. Data transformatons are handled internally so that all returned gradients are in standard observation space (i.e not log space etc).

```python
# load in the model you wish to use
from galaxygrad import QUASAR_ScoreNet72
prior = QUASAR_ScoreNet72
```
