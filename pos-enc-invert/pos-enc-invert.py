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

def sin_cos_pos_enc(pos: jax.Array, d_model: int, max_len: int):
 e = jnp.empty((d_model,))
 x = pos / (max_len ** jnp.arange(d_model / 2) / d_model)

 e = e.at[0::2].set(jnp.sin(x))
 e = e.at[1::2].set(jnp.cos(x))

 return e

sin_cos_pos_enc_batched = jax.vmap(sin_cos_pos_enc, (0, None, None), 0)

def normal_pos_enc(pos: jax.Array, means: jax.Array, variances: jax.Array):
 return jnp.array([jax.scipy.stats.norm.pdf(pos, loc=means[i], scale=variances[i]) for i in range(d_model)])

normal_pos_enc_batched = jax.vmap(normal_pos_enc, (0, None, None), 0)

# Generates a (randomly initialized) normal distribution position encoding
# Each dimension of the encoding is a probability from a random univariate
# gaussian. The distributions are scaled by the length and the means
# spaced across the length in order to give good differentiation
# between locations
def normal_pos_enc_arr(length: int, depth: int, key: jax.Array) -> jax.Array:
 interval = float(length) / float(depth)

 means = jnp.arange(depth) * interval
 print(f"{means=}")
 variances = jax.random.normal(key, (depth,)) * length
 print(f"{variances=}")

 position_probs: list[jax.Array] = list()
 for pos in range(length):
  position_probs.append(jax.scipy.stats.norm.pdf(pos, loc=means, scale=variances))

 return jnp.stack(position_probs)# (length, depth)
  
 

# class PosEnc:
#  def __call__(self, pos: int) -> jax.Array:
#   raise NotImplemented

# @dataclass
# class SinCosPosEnc(PosEnc):
#  d_model: int
#  max_len: int

#  def __call__(self, pos: int):
#   e = jnp.empty((self.d_model,))
#   x = pos / (self.max_len ** jnp.arange(self.d_model / 2) / self.d_model)

#   e = e.at[0::2].set(jnp.sin(x))
#   e = e.at[1::2].set(jnp.cos(x))

#   return e

# class NormalPosEnc(PosEnc):
#  d_model: int

#  def __init__(self, key: jax.Array, d_model: int, max_len: int):
#   self.d_model = d_model
  
#   k0, k1 = jr.split(key, 2)
#   self.means = jr.normal(k0, (d_model,)) * max_len
#   self.variances = jr.normal(k1, (d_model,)) * max_len

#   # self.sigma = jr.normal(k1, (d_model, d_model)) * max_len
#   # self.sigma = jnp.matmul(self.sigma.transpose(), self.sigma)

#   # print(f"mu: {self.mu}")
#   # print(f"sigma: {self.sigma}")

#  def __call__(self, pos: int):
#   x = jnp.repeat(pos, self.d_model)
#   probs = [ jax.scipy.stats.norm.pdf(pos, loc=self.means[i], scale=self.variances[i]) for i in range(self.d_model) ]

#   return jnp.array(probs)

class Model(nnx.Module):
 def __init__(self, *, in_features: int, rngs: nnx.Rngs):
  self.linear = nnx.Linear(in_features=in_features, out_features=1, rngs=rngs)

 def __call__(self, x: jax.Array) -> jax.Array:
  print(f"model call {x.shape=}")
  return nnx.sigmoid(self.linear(x))[:, 0]

@nnx.jit
def train_step(model: Model, optimizer, x: jax.Array, y: jax.Array):
 print(f"train_step: {x.shape =} {y.shape =}")
 def loss_fn(model: Model):
   y_pred = model(x)
   return optax.losses.squared_error(y_pred, y).mean()

 loss, grads = nnx.value_and_grad(loss_fn)(model)
 optimizer.update(grads)

 return loss

if __name__=="__main__":
 key = jr.key(493893)

 d_model = 32
 max_len = 100
 iters = 1000
 # batch = 10

 encodings = {
  'sincos': sin_cos_pos_enc_arr(max_len, d_model),
  'norm': normal_pos_enc_arr(max_len, d_model, key),
 }

 rngs = nnx.Rngs(0)

 # Models trying to recover the encoded position
 models = {
  'sincos': Model(in_features=d_model, rngs=rngs),
  'norm': Model(in_features=d_model, rngs=rngs)
 }

 optimizers = dict()
 for k,v in models.items():
  optimizers[k] = nnx.Optimizer(v, optax.adam(1e-3))

 total_losses: dict[str,float] = defaultdict(float)
 count = 0

 def mean_loss(name: str):
  return total_losses[name] / count

 x = jnp.arange(max_len) / max_len
 print(f"{ x.shape = }")

 # x = jax.nn.one_hot(jnp.arange(max_len), max_len)
 
 for _ in range(iters):
  count += 1
  for name, enc in encodings.items():
   m = models[name]

   pe = enc[:max_len, :]
   print(f"{pe.shape=}")

   # pred = m(enc).reshape((max_len,))

   # print(f"{ pred.shape = }")

   loss = train_step(m, optimizers[name], pe, x)
   print(f"{name} {loss=}")
   total_losses[name] += loss
   print(f"{name} total loss {total_losses[name]}")

   print(f"{name} mean loss: {mean_loss(name)}")

   
   
   
  # for pos in range(max_len):
  #  count += 1
  #  scaled_pos = jnp.array(pos / max_len)
  #  for name, model in models.items():
  #   enc = encodings[name]
  #   opt = optimizers[name]

  #   e = enc(pos)
  #   pos_again = model(e)

  #   loss = train_step(model, opt, scaled_pos, pos_again)
  #   total_losses[name] += loss

  #   print(f"{name} mean loss: {mean_loss(name)}")

    
 
