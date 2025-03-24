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

def linear_pos_enc_arr(*, length: int, weights: jax.Array, bias: jax.Array) -> jax.Array:
 y = jnp.arange(length) / length
 y = y.reshape((length, 1))

 return y * weights + bias

def default_linear_pos_enc_arr(*, length: int, depth: int, key: jax.Array, init: str) -> jax.Array:
 k0, k1 = jr.split(key, 2)

 if init == 'normal':
  w = jax.random.normal(k0, (depth,))
  b = jax.random.normal(k1, ())
 elif init == 'uniform':
  w = jax.random.uniform(k0, (depth,), minval=-1.0, maxval=1.0)
  b = jax.random.uniform(k1, (), minval=-1.0, maxval=1.0)
 else:
  raise ValueError()

 return linear_pos_enc_arr(length=length, weights=w, bias=b)

def normal_pos_enc_arr(*, length: int, depth: int, means: jax.Array, variances: jax.Array) -> jax.Array:
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
def default_normal_pos_enc_arr(*, length: int, depth: int, key: jax.Array) -> jax.Array:
 k0, k1 = jr.split(key, 2)

 means = jax.random.normal(k0, (depth))
 variances = jnp.abs(jax.random.normal(k1, (depth,)))

 return normal_pos_enc_arr(length=length, depth=depth, means=means, variances=variances)


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

# Randomly-parameterized, learning version of normal_pos_enc_arr2
class NormPosEnc(nnx.Module):
 def __init__(self, *, length: int, depth: int, rngs: nnx.Rngs):
  self.length = length
  self.depth = depth
  self.means = nnx.Param(jax.random.normal(rngs(), (depth,)))
  self.variances = nnx.Param(jnp.abs(jax.random.normal(rngs(), (depth,))))

 def pos_enc_arr(self) -> jax.Array:
  return normal_pos_enc_arr(
   length=self.length,
   depth=self.depth,
   means=self.means.value,
   variances=self.variances.value,
  )

# A learned densely-connected position encoding
class LinearPosEnc(nnx.Module):
 def __init__(self, *, length: int, depth: int, rngs: nnx.Rngs, init: str):
  self.length = length

  if init == 'normal':
   self.weights = nnx.Param(jax.random.normal(rngs(), (depth,)))
   self.bias = nnx.Param(jax.random.normal(rngs(), ()))
  elif init == 'uniform':
   self.weights = nnx.Param(jax.random.uniform(rngs(), (depth,), minval=-1.0, maxval=1.0))
   self.bias = nnx.Param(jax.random.uniform(rngs(), (), minval=-1.0, maxval=1.0))
  else:
   raise ValueError()
   

 def pos_enc_arr(self) -> jax.Array:
  return linear_pos_enc_arr(length=self.length, weights=self.weights, bias=self.bias)

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

class InvertedLinearPosEnc(nnx.Module):
 def __init__(self, *, length: int, depth: int, rngs: nnx.Rngs, init: str):
  self.length = length
  self.enc = LinearPosEnc(length=length, depth=depth, rngs=rngs, init=init)
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


@nnx.jit
def train_step_learn_normenc(*, model: InvertedNormPosEnc, opt: nnx.Optimizer, y: jax.Array):
 def loss_fn(model: InvertedNormPosEnc):
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

def default_opt():
 return optax.adam(1e-3)

def eprint(s: str):
 sys.stderr.write(s)
 sys.stderr.write('\n')

if __name__=="__main__":
 inits = 100
 d_model = 32
 max_len = 100
 iters = 20000
 # Whether to calculate variances in addition to means
 generate_var = False

 eprint(f"{inits=}")
 eprint(f"{d_model=}")
 eprint(f"{max_len=}")
 eprint(f"{iters=}")
 eprint(f"{generate_var=}")

 # Store losses so we can compute statistics
 # iter -> encoding name -> list of losses across all initializations
 iteration_losses = [ defaultdict(list) for _ in range(iters) ]
  
 for init in range(inits):
  eprint(f"{init=}")
  rngs = nnx.Rngs(init)

  encodings = {
   'sincos': sin_cos_pos_enc_arr(max_len, d_model),
   'normal': default_normal_pos_enc_arr(length=max_len, depth=d_model, key=rngs()),
   'direct1': direct_pos_enc(length=max_len, depth=d_model),
   'directN': direct_all_pos_enc(length=max_len, depth=d_model),
   'linear_normal': default_linear_pos_enc_arr(length=max_len, depth=d_model, key=rngs(), init='normal'),
   'linear_uniform': default_linear_pos_enc_arr(length=max_len, depth=d_model, key=rngs(), init='uniform'),
  }

  # Models trying to recover the encoded position
  models = { name: Inverter(in_features=d_model, rngs=rngs) for name in encodings.keys() }

  inverted_encodings = {
   'normal_learned': InvertedNormPosEnc(length=max_len, depth=d_model, rngs=rngs),
   'linear_normal_learned': InvertedLinearPosEnc(length=max_len, depth=d_model, rngs=rngs, init='normal'),
   'linear_uniform_learned': InvertedLinearPosEnc(length=max_len, depth=d_model, rngs=rngs, init='uniform'),
  }

  train_steps = {
   'normal_learned': train_step_learn_normenc,
   'linear_normal_learned': train_step_learn_linear,
   'linear_uniform_learned': train_step_learn_linear,
  }

  optimizers = { k: nnx.Optimizer(v, default_opt()) for k,v in models.items() }
  for name, inv_enc in inverted_encodings.items():
   optimizers[name] = nnx.Optimizer(inv_enc, default_opt())

  names_set = set(encodings.keys())
  names_set.update(inverted_encodings.keys())
  names = list(names_set)
  names.sort()

  def record_loss(iter: int, name: str, loss: jax.Array):
   iter_losses = iteration_losses[iter]
   loss_ = float(loss)

   if generate_var:
    iter_losses[name].append(loss_)
   else:
    # Just keep a running sum so we can calculate means
    if len(iter_losses[name]) == 0:
     iter_losses[name] = [loss_]
    else:
     iter_losses[name][0] += loss_

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
  return math.fsum(iteration_losses[i][name]) / iters

 def var_iter_loss(i: int, name: str) -> float:
  mean = mean_iter_loss(i, name)

  var = 0.0

  for l in iteration_losses[i][name]:
   var += (mean - l)**2

  return var 

 # Example:
 # iter,norm,norm2,norm_learned,norm2_learned
 # 0,0.3,0.2,0.25,0.4
 # 1,0.29,0.19,0.24,0.39
 # 2,0.28,0.18,0.23,0.38
 sys.stdout.write('iter')
 for n in names:
  if generate_var:
   sys.stdout.write(f",{n} (mean)")
   sys.stdout.write(f",{n} (var)")
  else:
   sys.stdout.write(f",{n}")
 sys.stdout.write('\n')

 for i in saved_iters:
  sys.stdout.write(str(i))
  
  for name in names:
   sys.stdout.write(',')
   sys.stdout.write(str(mean_iter_loss(i, name)))
   if generate_var:
    sys.stdout.write(',')
    sys.stdout.write(str(var_iter_loss(i, name)))
  sys.stdout.write('\n')
