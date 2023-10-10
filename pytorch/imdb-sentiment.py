from collections import Counter
import itertools
import json
import os

from pytorch_lightning import LightningModule, Trainer
from torchmetrics import Accuracy, AUROC, F1Score

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

EMBEDDING_DIMS = 20
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
  doc_emb = emb.sum(1)# sum over the word indices
  out1 = self.linear1(doc_emb)
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
 path = os.environ['HOME'] + "/Data/com/github/nas5w/imdb-data/reviews.json"

 with open(path) as r:
  raw = json.load(r)

 
 vocab = set()
 vocab.add("__padding__")
 for datum in raw:
  words = datum['t'].split()
  vocab.update(words)

 # print(vocab)
 print(f"vocab_len: {len(vocab)}")

 word_to_idx = {word: i for i, word in enumerate(vocab)}
 padding_idx = word_to_idx["__padding__"]

 indexed = list()
 lens = Counter()

 for datum in raw:
  words = datum['t'].split()
  word_indices = [ word_to_idx[word] for word in words ]

  lens[len(word_indices)] += 1

  class_ = datum['s']
  indexed .append((word_indices, class_))
 
 del raw

 sorted_lens = list(lens.items())
 sorted_lens.sort(key = lambda x: x[0])
 cum = 0
 target_len = None
 for l, n in sorted_lens:
  cum += n
  pct = cum / lens.total()
  if pct >= 0.95:
   target_len = l
   break

 print(f"target_len: {target_len}")
 print(f"padding_idx: {padding_idx}")

 data = list()
 for x, y in indexed:
  # Pad to target_len
  if len(x) < target_len:
   x.extend([padding_idx] * (target_len - len(x)))
  else:
   x = x[:target_len]

  data.append((torch.tensor(x, dtype=iX, device=device), torch.tensor(y, dtype=fX, device=device)))

 del indexed
 
 dataset = ObjectDataset(data)

 

 train, val, test = random_split(dataset, [0.8, 0.1, 0.1])

 # print(dir(train))

 train_loader = DataLoader(train, shuffle=True, batch_size=100)
 val_loader = DataLoader(val, batch_size=100)
 test_loader = DataLoader(test, batch_size=100)

 model = ImdbSentiment(len(vocab))

 trainer = Trainer(max_epochs=50)
 trainer.fit(model, train_loader, val_loader)

 trainer.test(ckpt_path='best', dataloaders=test_loader)
