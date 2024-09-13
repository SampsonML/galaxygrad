[![arXiv](https://img.shields.io/badge/arXiv-2401.07313-<COLOR>.svg)](https://arxiv.org/abs/2401.07313)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI](https://img.shields.io/pypi/v/galaxygrad?label=pypi%20package)](https://pypi.org/project/galaxygrad/)
[![PyPI - Downloads (https://img.shields.io/pypi/dm/galaxygrad)](https://pypistats.org/packages/galaxygrad)
# GalaxyGrad
Package for a score-based diffusion model trained on HSC galaxies, ZTF like simulations, and lensed quasars.
See https://pypi.org/project/galaxygrad/0.1.8/
Usage:
Install the package from pip

```shell
pip install galaxygrad
```
You can now use the pre-loaded priors on and 2D arrays the same size as the numerical value of the prior name, ie HSC_ScoreNet64 takes 64 by 64 arrays.

```python
# load in the model you wish to use
from galaxygrad import HSC_ScoreNet64 # current options are {HSC_ScoreNet32, HSC_ScoreNet64, ZTF_ScoreNet32, ZTF_ScoreNet64, QUASAR_ScoreNet72}
prior = HSC_ScoreNet64

galaxy = np.ones([64,64])
galaxy = np.expand_dims(galaxy, axis=0) # the prior requires 3 dimensions for easier use in vmapped functions (batch processing)
gradients = prior(galaxy)
```

You can also directly adjust the desired temperature each time
```python
# load in the model you wish to use
gradients = HSC_ScoreNet64(galaxy,t=0.05)
```
For adjusting the temperature of the model to a new default value (useful when running inside optimization routines) the cleanest way is to instantiate a prior class and call the prior through this
```python
# define a class for temperature adjustable prior
class TempScore:
    """Temperature adjustable ScorePrior"""
    def __init__(self, model, temp=0.02):
        self.model = model
        self.temp = temp
    def __call__(self, x):
        return self.model(x, t=self.temp)
```
Now you may call the prior through the class with any custom temperature between 0 --> 10, though nothing above 0.1 would be reccomended.
```python
temp = 0.02
prior = TempScore(model=HSC_ScoreNet64, temp=temp)
gradients = prior(galaxy)
```
This method also removes any parameters that one may not like to trace.
