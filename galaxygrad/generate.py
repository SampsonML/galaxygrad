# Generate galaxy samples via integrating SDE with ScoreNet gradients

from .scorenet import HSC_ScoreNet32, HSC_ScoreNet64
import einops
import jax
import functools as ft
import diffrax as dfx
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

t1 = 10.0
int_beta = lambda t: t
weight = lambda t: 1 - jnp.exp(-int_beta(t))


@eqx.filter_jit
def single_sample_fn(model, int_beta, data_shape, dt0, t1, key):
    def drift(t, y, args):
        _, beta = jax.jvp(int_beta, (t,), (jnp.ones_like(t),))
        return -0.5 * beta * (y + model(y, t))

    term = dfx.ODETerm(drift)
    solver = dfx.Tsit5()
    t0 = 0.0
    y1 = jr.normal(key, data_shape)
    # reverse time, solve from t1 to t0
    # sol = dfx.diffeqsolve(term, solver, t1, t0, -dt0, y1, adjoint=dfx.NoAdjoint())
    sol = dfx.diffeqsolve(term, solver, t1, t0, -dt0, y1)
    return sol.ys[0]


def generateSamples(n_samples=1, hi_res=True, seed=1992):
    if hi_res:
        model = HSC_ScoreNet64
        data_shape = (1, 64, 64)
        data_mean = 0.017
        data_std = 0.09
    else:
        model = HSC_ScoreNet32
        data_shape = (1, 32, 32)
        data_mean = 0.023
        data_std = 0.11

    # create samples
    key = jr.PRNGKey(seed)
    # data = y1 = jr.normal(key, data_shape)
    sample_size = n_samples
    dt0 = 0.05  # sample step size
    sample_key = jr.split(key, sample_size**2)
    sample_fn = ft.partial(single_sample_fn, model, int_beta, data_shape, dt0, t1)
    sample = jax.vmap(sample_fn)(sample_key)
    # sample = data_mean + data_std * sample
    # sample = jnp.clip(sample, 0, 1)
    # sample = einops.rearrange(
    #    sample, "(n1 n2) 1 h w -> (n1 h) (n2 w)", n1=sample_size, n2=sample_size
    # )

    return sample
