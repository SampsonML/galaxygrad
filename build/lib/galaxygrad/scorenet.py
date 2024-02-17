import jax.random as jr
import equinox as eqx
from .models_eqx import ScoreNet
import os
import jax.numpy as jnp

# model params
patch_size=4
hidden_size=64
mix_patch_size=512
mix_hidden_size=512
num_blocks=4
t1=10.0

# noise -- time schedule
int_beta = lambda t: t  

# rng numbers 
key = jr.PRNGKey(42)
model_key, train_key, loader_key, sample_key = jr.split(key, 4)

# initialise model for 64 res
data_shape = (1, 64, 64)
model64 = ScoreNet(
        data_shape,
        patch_size,
        hidden_size,
        mix_patch_size,
        mix_hidden_size,
        num_blocks,
        t1,
        key=model_key,
)

# load 64 res model
FN = os.path.join(os.path.dirname(__file__), 'eqx_hsc_ScoreNet64.eqx')
HSC_64 = eqx.tree_deserialise_leaves(FN, model64)
def model_wrapper(x, t=0.01):
    sigma = 0.1
    x_ = jnp.log(x + 1) / sigma
    raw_grad = HSC_64(x_, t)
    transform_grad = raw_grad * ( 1 / ( sigma * (x + 1) )) # analytic derrivitive    
    return transform_grad

# define HSC trained 64 by 64 score function
HSC_ScoreNet64_func = model_wrapper

# define a class to store shape attribute
class HSC_Scorenet64:
    def __init__(self):
        self.shape = (64,64)
    def __call__(self, x, t=0.02):
        return HSC_ScoreNet64_func(x, t)
HSC_ScoreNet64 = HSC_Scorenet64()

# initialise model for 32 res
data_shape = (1, 32, 32)
model32 = ScoreNet(
        data_shape,
        patch_size,
        hidden_size,
        mix_patch_size,
        mix_hidden_size,
        num_blocks,
        t1,
        key=model_key,
)

# load 32 res model
FN = os.path.join(os.path.dirname(__file__), 'eqx_hsc_ScoreNet32.eqx')
HSC_32 = eqx.tree_deserialise_leaves(FN, model32)
def model_wrapper(x, t=0.01):
    sigma = 0.1
    x_ = jnp.log(x + 1) / sigma
    raw_grad = HSC_32(x_, t)
    transform_grad = raw_grad * ( 1 / ( sigma * (x + 1) )) # analytic derrivitive    
    return transform_grad
HSC_ScoreNet32_func = model_wrapper

# define a class to store shape attribute
class HSC_Scorenet32:
    def __init__(self):
        self.shape = (32,32)
    def __call__(self, x, t=0.02):
        return HSC_ScoreNet32_func(x, t)
HSC_ScoreNet32 = HSC_Scorenet32()

# initialise model for 32 res
data_shape = (1, 32, 32)
model32 = ScoreNet(
        data_shape,
        patch_size,
        hidden_size,
        mix_patch_size,
        mix_hidden_size,
        num_blocks,
        t1,
        key=model_key,
)

# load 32 res model
FN = os.path.join(os.path.dirname(__file__), 'eqx_ZTF_ScoreNet32.eqx')
ZTF_32 = eqx.tree_deserialise_leaves(FN, model32)
def model_wrapper(x, t=0.01):
    sigma = 0.1
    x_ = jnp.log(x + 1) / sigma
    raw_grad = ZTF_32(x_, t)
    transform_grad = raw_grad * ( 1 / ( sigma * (x + 1) )) # analytic derrivitive    
    return transform_grad
ZTF_ScoreNet32_func = model_wrapper

# define a class to store shape attribute
class ZTF_Scorenet32:
    def __init__(self):
        self.shape = (32,32)
    def __call__(self, x, t=0.02):
        return ZTF_ScoreNet32_func(x, t)
ZTF_ScoreNet32 = ZTF_Scorenet32()

# initialise model for 64 res
data_shape = (1, 64, 64)
model32 = ScoreNet(
        data_shape,
        patch_size,
        hidden_size,
        mix_patch_size,
        mix_hidden_size,
        num_blocks,
        t1,
        key=model_key,
)

# load 32 res model
FN = os.path.join(os.path.dirname(__file__), 'eqx_ZTF_ScoreNet64.eqx')
ZTF_64 = eqx.tree_deserialise_leaves(FN, model32)
def model_wrapper(x, t=0.01):
    sigma = 0.1
    x_ = jnp.log(x + 1) / sigma
    raw_grad = ZTF_64(x_, t)
    transform_grad = raw_grad * ( 1 / ( sigma * (x + 1) )) # analytic derrivitive    
    return transform_grad
ZTF_ScoreNet64_fun = model_wrapper

# define a class to store shape attribute
class ZTF_Scorenet64:
    def __init__(self):
        self.shape = (64,64)
    def __call__(self, x, t=0.02):
        return ZTF_ScoreNet64_func(x, t)
ZTF_ScoreNet64 = ZTF_Scorenet64()

