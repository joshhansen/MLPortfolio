from imdb_sa_common import load

from collections import Counter
import itertools
import json
import os
import re

from pytorch_lightning import LightningModule, Trainer
from torchmetrics import Accuracy, AUROC, F1Score

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

EMBEDDING_DIMS = 20
ATTN_DIMS=5
fX = torch.float64
iX = torch.long
device = torch.device("cuda")

class ObjectDataset(Dataset):
 def __init__(self, obj):
  self.obj = obj

 def __getitem__(self, i):
  return self.obj[i]

 def __len__(self):
  return len(self.obj)

class ImdbSentiment(LightningModule):
 def __init__(self, vocab_size):
  super().__init__()
  self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIMS, dtype=fX)
  self.attn = nn.MultiheadAttention(EMBEDDING_DIMS, ATTN_DIMS, dtype=fX)
  self.linear1 = nn.Linear(EMBEDDING_DIMS, EMBEDDING_DIMS // 2, dtype=fX)
  self.linear2 = nn.Linear(EMBEDDING_DIMS // 2, 1, dtype=fX)

  self.acc = Accuracy(task='binary').to(device)
  self.auroc = AUROC(task='binary').to(device)
  self.f1score = F1Score(task='binary').to(device)

  self.metrics = [
   self.acc, self.auroc, self.f1score
  ]

 def params(self):
  return itertools.chain(self.embedding.parameters(), self.linear1.parameters(), self.linear2.parameters())

 def forward(self, x, y):
  emb = self.embedding(x)
  attn = self.attn(emb)
  out1 = self.linear1(attn)
  out2 = self.linear2(out1)

  # print(f"emb {emb.shape} {emb.dtype}")
  # print(f"doc_emb {doc_emb.shape} {doc_emb.dtype}")
  # print(f"out1 {out1.shape} {out1.dtype}")
  # print(f"out2 {out2.shape} {out2.dtype}")

  return torch.nn.functional.sigmoid(out2)

 def training_step(self, batch, batch_idx):
  inputs, target = batch
  # output = self(inputs, target)
  output = self.forward(inputs, target)
  # print(f"out shape: {output.shape} {output.dtype}")
  # print(f"target shape: {target.shape} {target.dtype}")

  target_resized = target.view(output.shape)
  loss = torch.nn.functional.binary_cross_entropy(output, target_resized)

  self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

  # Make specific predictions
  preds = output.round()

  for m in self.metrics:
   mval = m(preds, target_resized)
   self.log(f"test_{m.__class__.__name__}", mval, on_step=True, on_epoch=True, prog_bar=True)

  return loss

 def validation_step(self, batch, batch_idx):
  inputs, target = batch
  output = self.forward(inputs, target)

  target_resized = target.view(output.shape)
  loss = torch.nn.functional.binary_cross_entropy(output, target_resized)

  self.log('test_loss', loss, prog_bar=True)

  # Make specific predictions
  preds = output.round()

  for m in self.metrics:
   mval = m(preds, target_resized)
   self.log(f"test_{m.__class__.__name__}", mval, prog_bar=True)

  return loss

 def test_step(self, batch, batch_idx):
  inputs, target = batch
  output = self.forward(inputs, target)

  target_resized = target.view(output.shape)
  loss = torch.nn.functional.binary_cross_entropy(output, target_resized)

  # Make specific predictions
  preds = output.round()

  self.log('test_loss', loss, prog_bar=True)

  for m in self.metrics:
   mval = m(preds, target_resized)
   self.log(f"test_{m.__class__.__name__}", mval, prog_bar=True)

  return loss
  

 def configure_optimizers(self):
    return torch.optim.Adam(self.params(), lr=1e-3)

if __name__=="__main__":
 data = load(False)

 def tensorize(data):
  def tensor(datum):
   x, y = datum
   return (torch.tensor(x, dtype=iX, device=device), torch.tensor(y, dtype=fX, device=device))
  return list(map(tensor, data))

 train = tensorize(data['train'])
 val = tensorize(data['val'])
 test = tensorize(data['test'])
 vocab_len = data['vocab_len']

 train_loader = DataLoader(train, shuffle=True, batch_size=100)
 val_loader = DataLoader(val, batch_size=100)
 test_loader = DataLoader(test, batch_size=100)

 model = ImdbSentiment(vocab_len)

 trainer = Trainer(max_epochs=50)
 trainer.fit(model, train_loader, val_loader)

 trainer.test(ckpt_path='best', dataloaders=test_loader)
