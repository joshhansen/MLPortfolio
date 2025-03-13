# Can a dense neural net recover the input to a multivariate normal pdf?
from collections import defaultdict
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
import optax

# Generates a length x depth matrix where each row is a depth-sized vector of position encodings
# corresponding to the row index
def sin_cos_pos_enc_arr(length: int, depth: int) -> jax.Array:
 if depth % 2 != 0:
  raise ValueError(f"depth must be even but { depth = }")

 depth = depth // 2

 positions = jnp.arange(length)[:, jnp.newaxis]        # (seq, 1)
 depths = jnp.arange(depth)[jnp.newaxis, :]/depth      # (1, depth/2)

 angle_rates = 1 / (10000**depths.astype(jnp.float32)) # (1, depth/2)
 # = 1 / 10000**(2i / d_model) in the paper
 
 angle_rads = positions * angle_rates                  # (seq, depth/2)
 # = pos / 10000**(2i / d_model) in the paper

 s = jnp.sin(angle_rads)
 c = jnp.cos(angle_rads)

 return jnp.concatenate([s, c], axis=-1)# (seq, depth)

# Generates a (randomly initialized) normal distribution position encoding
# Each dimension of the encoding is a probability from a random univariate
# gaussian. The distributions are scaled by the length and the means
# spaced across the length in order to give good differentiation
# between locations
def normal_pos_enc_arr(length: int, depth: int, key: jax.Array) -> jax.Array:
 interval = float(length) / float(depth)
 # interval = 100

 means = jnp.arange(depth) * interval
 # variances = jax.random.normal(key, (depth,)) * length
 # variances = (jnp.arange(depth) + 1).astype(jnp.float32)
 variances = jnp.ones((depth,))

 position_probs: list[jax.Array] = list()
 for pos in range(length):
  position_probs.append(jax.scipy.stats.norm.pdf(pos, loc=means, scale=variances))

 return jnp.stack(position_probs)# (length, depth)
 
class Model(nnx.Module):
 def __init__(self, *, in_features: int, rngs: nnx.Rngs):
  self.linear = nnx.Linear(in_features=in_features, out_features=1, rngs=rngs)

 def __call__(self, x: jax.Array) -> jax.Array:
  return nnx.sigmoid(self.linear(x))[:, 0]

@nnx.jit
def train_step(model: Model, optimizer, x: jax.Array, y: jax.Array):
 def loss_fn(model: Model):
   y_pred = model(x)
   return optax.losses.squared_error(y_pred, y).mean()

 loss, grads = nnx.value_and_grad(loss_fn)(model)
 optimizer.update(grads)

 return loss

if __name__=="__main__":
 inits = 20
 d_model = 32
 max_len = 100
 batch = 1000
 print(f"{inits=}")
 print(f"{d_model=}")
 print(f"{max_len=}")
 print(f"{batch=}")

 for init in range(inits):
  key = jr.key(init)
  k0, k1 = jr.split(key, 2)

  encodings = {
   'sincos': sin_cos_pos_enc_arr(max_len, d_model),
   'norm': normal_pos_enc_arr(max_len, d_model, k0),
  }

  rngs = nnx.Rngs(k1)

  # Models trying to recover the encoded position
  models = {
   'sincos': Model(in_features=d_model, rngs=rngs),
   'norm': Model(in_features=d_model, rngs=rngs)
  }

  optimizers = dict()
  for k,v in models.items():
   optimizers[k] = nnx.Optimizer(v, optax.adam(1e-1))

  total_losses: dict[str,float] = defaultdict(float)
  count = 0

  def mean_loss(name: str):
   return total_losses[name] / count

  x = jnp.arange(max_len) / max_len

  for count in range(batch):
   for name, enc in encodings.items():
    m = models[name]

    pe = enc[:max_len, :]

    loss = train_step(m, optimizers[name], pe, x)
    total_losses[name] += loss

  for name in encodings.keys():
   print(f"{init=} {name} mean loss: {mean_loss(name)}")
