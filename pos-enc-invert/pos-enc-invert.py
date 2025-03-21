# Can a dense neural net recover the input to a multivariate normal pdf?
from collections import defaultdict
from dataclasses import dataclass
import math
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
import optax

# jax.config.update("jax_debug_nans", True)

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
#
# Deprecated
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

def normal_pos_enc_arr2(*, length: int, depth: int, means: jax.Array, variances: jax.Array) -> jax.Array:
  samples: list[jax.Array] = list()
  samples.append(jnp.repeat(-9999999, depth))

  # print(f"means: {self.means}")
  # print(f"variances: {self.variances}")
  # print(f"Initial {samples=}")

  # avoid referencing probability 0 or 1, which have -inf and +inf as outputs
  for pos in range(1, length):
   probs = jnp.repeat(pos / length, depth)
   # print(f"{probs=}")
   norm_samples = inverse_norm_cdf(probs=probs, means=means, variances=variances)
   samples.append(norm_samples)

  return jnp.stack(samples)

# Generates a (randomly initialized) normal distribution position encoding
#
# Each row of the encoding represents a sequence position in a sequence of
# the specified length.
#
# Each row consists of `depth` components, each a "sample" from a corresponding
# univariate gaussian. ("Sample" in quotes because the values are not random.)
# Though the values are deterministic, they are normally distributed on a per-
# depth-dimension basis.
def default_normal_pos_enc_arr2(*, length: int, depth: int, key: jax.Array) -> jax.Array:
 k0, k1 = jr.split(key, 2)

 means = jax.random.normal(k0, (depth))
 
 # interval = float(length) / float(depth)
 # means = jnp.arange(depth) * interval

 # Center the means around 0
 # means -= depth/2

 

 # variances = jax.random.normal(key, (depth,)) * length
 # variances = (jnp.arange(depth) + 1).astype(jnp.float32)
 # variances = jnp.ones((depth,))
 variances = jnp.abs(jax.random.normal(k1, (depth,)))

 return normal_pos_enc_arr2(length=length, depth=depth, means=means, variances=variances)


# This says how far in the normal distribution with given parameters you have to integrate
# to to get a given input probability
#
# Note: we use the absolute value of the variances to avoid sqrt of negative
def inverse_norm_cdf(*, probs: jax.Array, means: jax.Array, variances: jax.Array) -> jax.Array:
 std_norm = jax.scipy.special.ndtri(probs)
 root_variances = jnp.sqrt(jnp.abs(variances))
 return jnp.multiply(std_norm, root_variances) + means
 
# A "position encoding" that represents the position as a float between 0 and 1 in the first
# dimension; the remaining dimensions are 0
def direct_pos_enc(*, length: int, depth: int) -> jax.Array:
 enc = jnp.zeros((length, depth), dtype=jnp.float32)
 positions = jnp.arange(length) / length
 return enc.at[:, 0].set(positions)

# A "position encoding" that represents the position as a float between 0 and 1 in all dimensions
def direct_all_pos_enc(*, length: int, depth: int) -> jax.Array:
 positions = jnp.arange(length) / length
 # (seq,)

 enc = jnp.tile(positions, (depth, 1))
 # (depth, seq)

 return enc.transpose()
 # (seq, depth)

# Randomly-parameterized, learning version of normal_pos_enc_arr
class NormPosEnc(nnx.Module):
 def __init__(self, *, length: int, depth: int, rngs: nnx.Rngs):
  self.length = length
  self.depth = depth
  self.means = nnx.Param(jax.random.normal(rngs(), (depth,)))
  self.variances = nnx.Param(jax.random.normal(rngs(), (depth,)))

 def pos_enc_arr(self) -> jax.Array:
  position_probs: list[jax.Array] = list()
  for pos in range(self.length):
   position_probs.append(jax.scipy.stats.norm.pdf(pos, loc=self.means.value, scale=self.variances.value))

  return jnp.stack(position_probs)# (length, depth)
  
# Randomly-parameterized, learning version of normal_pos_enc_arr2
class NormPosEnc2(nnx.Module):
 def __init__(self, *, length: int, depth: int, rngs: nnx.Rngs):
  self.length = length
  self.depth = depth
  self.means = nnx.Param(jax.random.normal(rngs(), (depth,)))
  self.variances = nnx.Param(jnp.abs(jax.random.normal(rngs(), (depth,))))

 def pos_enc_arr(self) -> jax.Array:
  return normal_pos_enc_arr2(
   length=self.length,
   depth=self.depth,
   means=self.means.value,
   variances=self.variances.value,
  )

# A learned densely-connected position encoding
class LinearPosEnc(nnx.Module):
 def __init__(self, *, length: int, depth: int, rngs: nnx.Rngs):
  self.length = length
  self.ff = nnx.Linear(in_features=1, out_features=depth, rngs=rngs)

 def pos_enc_arr(self) -> jax.Array:
  y = jnp.arange(self.length) / self.length
  y = y.reshape((self.length, 1))
  return self.ff(y)
  

class Inverter(nnx.Module):
 def __init__(self, *, in_features: int, rngs: nnx.Rngs):
  self.linear = nnx.Linear(in_features=in_features, out_features=1, rngs=rngs)

 def __call__(self, x: jax.Array) -> jax.Array:
  return nnx.sigmoid(self.linear(x))[:, 0]

class InvertedNormPosEnc(nnx.Module):
 def __init__(self, *, length: int, depth: int, rngs: nnx.Rngs):
  self.length = length
  self.enc = NormPosEnc(length=length, depth=depth, rngs=rngs)
  self.invert = Inverter(in_features=depth, rngs=rngs)

 def __call__(self):
  enc = self.enc.pos_enc_arr()
  y = enc[:self.length, :]
  y_pred = self.invert(y)
  return y_pred

class InvertedNormPosEnc2(nnx.Module):
 def __init__(self, *, length: int, depth: int, rngs: nnx.Rngs):
  self.length = length
  self.enc = NormPosEnc2(length=length, depth=depth, rngs=rngs)
  self.invert = Inverter(in_features=depth, rngs=rngs)

 def __call__(self):
  enc = self.enc.pos_enc_arr()
  # print(f"{enc=}")
  y = enc[:self.length, :]
  y_pred = self.invert(y)
  return y_pred

class InvertedLinearPosEnc(nnx.Module):
 def __init__(self, *, length: int, depth: int, rngs: nnx.Rngs):
  self.length = length
  self.enc = LinearPosEnc(length=length, depth=depth, rngs=rngs)
  self.invert = Inverter(in_features=depth, rngs=rngs)

 def __call__(self):
  enc = self.enc.pos_enc_arr()
  # print(f"{enc=}")
  y = enc[:self.length, :]
  y_pred = self.invert(y)
  return y_pred
 
  
@nnx.jit
def train_step(*, model: Inverter, opt: nnx.Optimizer, x: jax.Array, y: jax.Array):
 def loss_fn(model: Inverter):
   y_pred = model(x)
   return optax.losses.squared_error(y_pred, y).mean()

 loss, grads = nnx.value_and_grad(loss_fn)(model)
 opt.update(grads)

 return loss

# y: sequence positions / length
@nnx.jit
def train_step_learn_normenc(*, model: InvertedNormPosEnc, opt: nnx.Optimizer, y: jax.Array):
 def loss_fn(model: InvertedNormPosEnc):
  y_pred = model()
  return optax.losses.squared_error(y_pred, y).mean()
 
 loss, grads = nnx.value_and_grad(loss_fn)(model)
 opt.update(grads)

 return loss

@nnx.jit
def train_step_learn_normenc2(*, model: InvertedNormPosEnc2, opt: nnx.Optimizer, y: jax.Array):
 def loss_fn(model: InvertedNormPosEnc2):
  y_pred = model()
  l = optax.losses.squared_error(y_pred, y).mean()
  return l
 
 loss, grads = nnx.value_and_grad(loss_fn)(model)
 opt.update(grads)

 return loss

@nnx.jit
def train_step_learn_linear(*, model: InvertedLinearPosEnc, opt: nnx.Optimizer, y: jax.Array):
 def loss_fn(model: InvertedLinearPosEnc):
  y_pred = model()
  l = optax.losses.squared_error(y_pred, y).mean()
  return l
 
 loss, grads = nnx.value_and_grad(loss_fn)(model)
 opt.update(grads)

 return loss

# Should we record the results of a given training iteration?
def record_iter(i: int) -> bool:
 return True
 # try:
 #  l = math.log10(i)
 #  return int(l) == l
 # except ValueError:
 #  # Record iteration 0
 #  return True

def default_opt():
 return optax.adam(1e-3)

if __name__=="__main__":
 inits = 50
 d_model = 32
 max_len = 100
 iters = 1000
 print(f"{inits=}")
 print(f"{d_model=}")
 print(f"{max_len=}")
 print(f"{iters=}")

 # Store losses so we can compute statistics
 losses_by_iteration: dict[int, dict[str, list[float]]] = dict()
 def iteration_losses(i: int) -> dict[str, list[float]]:
  try:
   return losses_by_iteration[i]
  except KeyError:
   losses_by_iteration[i] = defaultdict(list)
   return losses_by_iteration[i]
  
 for init in range(inits):
  rngs = nnx.Rngs(init)

  encodings = {
   'sincos': sin_cos_pos_enc_arr(max_len, d_model),
   # 'norm': normal_pos_enc_arr(max_len, d_model, rngs()),
   'norm2': default_normal_pos_enc_arr2(length=max_len, depth=d_model, key=rngs()),
   'direct': direct_pos_enc(length=max_len, depth=d_model),
   'direct_all': direct_all_pos_enc(length=max_len, depth=d_model)
  }

  # Models trying to recover the encoded position
  models = { name: Inverter(in_features=d_model, rngs=rngs) for name in encodings.keys() }

  inverted_encodings = {
   # 'norm_inverted': InvertedNormPosEnc(length=max_len, depth=d_model, rngs=rngs),
   'norm_inverted2': InvertedNormPosEnc2(length=max_len, depth=d_model, rngs=rngs),
   'linear_inverted': InvertedLinearPosEnc(length=max_len, depth=d_model, rngs=rngs),
  }

  train_steps = {
   # 'norm_inverted': train_step_learn_normenc,
   'norm_inverted2': train_step_learn_normenc2,
   'linear_inverted': train_step_learn_linear,
  }

  optimizers = { k: nnx.Optimizer(v, default_opt()) for k,v in models.items() }
  for name, inv_enc in inverted_encodings.items():
   optimizers[name] = nnx.Optimizer(inv_enc, default_opt())

  names_set = set(encodings.keys())
  names_set.update(inverted_encodings.keys())
  names = list(names_set)
  names.sort()

  def record_loss(iter: int, name: str, loss: jax.Array):
   if record_iter(iter):
    iter_losses = iteration_losses(iter)
    loss_ = float(loss)
    iter_losses[name].append(loss_)

  x = jnp.arange(max_len) / max_len

  for count in range(iters):
   for name, enc in encodings.items():
    m = models[name]

    pe = enc[:max_len, :]

    loss = train_step(model=m, opt=optimizers[name], x=pe, y=x)
    record_loss(count, name, loss)

   for name, inv_enc in inverted_encodings.items():
    y = jnp.arange(max_len).astype(jnp.float32) / max_len
    opt = optimizers[name]
    step = train_steps[name]
    loss = step(model=inv_enc, opt=opt, y=y)
    record_loss(count, name, loss)
    
 saved_iters = list(losses_by_iteration.keys())
 saved_iters.sort()
 
 def mean_iter_loss(i: int, name: str) -> float:
  return math.fsum(iteration_losses(i)[name]) / iters

 def var_iter_loss(i: int, name: str) -> float:
  mean = mean_iter_loss(i, name)

  var = 0.0

  for l in iteration_losses(i)[name]:
   var += (mean - l)**2

  return var 

 # Example:
 # iter,norm,norm2,norm_learned,norm2_learned
 # 0,0.3,0.2,0.25,0.4
 # 1,0.29,0.19,0.24,0.39
 # 2,0.28,0.18,0.23,0.38
 sys.stdout.write('iter')
 for n in names:
  sys.stdout.write(f",{n} (mean)")
  sys.stdout.write(f",{n} (var)")
 sys.stdout.write('\n')

 for i in saved_iters:
  sys.stdout.write(str(i))
  
  for name in names:
   sys.stdout.write(',')
   sys.stdout.write(str(mean_iter_loss(i, name)))
   sys.stdout.write(',')
   sys.stdout.write(str(var_iter_loss(i, name)))
  sys.stdout.write('\n')
