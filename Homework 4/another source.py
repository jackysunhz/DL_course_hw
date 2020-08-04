# https://github.com/keitakurita/practical-torchtext/blob/master/Lesson%201%20intro%20to%20torchtext%20with%20text%20classification.ipynb


import torch
import torchtext
from torchtext.datasets import text_classification
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import Field
from torch.utils.data import DataLoader
import time
from torch.utils.data.dataset import random_split
import pandas as pd
import numpy as np
import re
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer

########################
# process csv file if using both training and validation set for inputs

def prepare_csv(val_ratio = 0.2):
    df_train = pd.read_csv("data/train.csv")
    idx = np.arange(df_train.shape[0])
    np.random.seed(42)
    np.random.shuffle(idx)
    val_size = int(len(idx) * val_ratio)
    df_train.iloc[idx[val_size:], :].to_csv('temp/d_train.csv', index=False)
    df_train.iloc[idx[:val_size], :].to_csv("temp/d_val.csv", index=False)

prepare_csv()

# preprocessing
tokenizer = lambda x: x.split()

txt_field = torchtext.data.Field(sequential=True, tokenize=tokenizer, include_lengths=True, use_vocab=True)
label_field = torchtext.data.Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)

train_val_fields = [
    ('labels', label_field),
    ('text', txt_field)
]

trainds, valds = torchtext.data.TabularDataset.splits(
path='temp/',train='d_train.csv',validation='d_val.csv', format='csv',skip_header=True,
fields=train_val_fields)

test_dataset = torchtext.data.TabularDataset(
path='data/test.csv', format='csv',skip_header=True,
fields=[('text', txt_field)])

# build vocabulary

txt_field.build_vocab(trainds, valds)
label_field.build_vocab(trainds)
vars(label_field.vocab)

# load data in batches
# BATCH_SIZE = 3


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

traindl, valdl = torchtext.data.BucketIterator.splits(datasets=(trainds, valds),
                                                      batch_sizes=(512,1024),
                                                      sort_key=lambda x: len(x.text),
                                                      device=device,
                                                      sort_within_batch=True,
                                                      repeat=False)


class BatchGenerator:
    def __init__(self, dl, x_field, y_field):
        self.dl, self.x_field, self.y_field = dl, x_field, y_field

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x_field)
            y = getattr(batch, self.y_field)
            yield (X, y)

train_batch_it = BatchGenerator(traindl, 'text', 'labels')
val_batch_it = BatchGenerator(valdl, 'text', 'labels')

train_batch_it.dl.dataset.examples[0].text

# len(traindl)
# len(valdl)

# temp = next(iter(traindl))
# temp.text

# return word indices and lengths
# batch.text

# batch.dataset.fields
# txt_field.vocab.itos[1]
# traindl.labels
# vars(traindl.dataset[0])

"""
def idxtosent(batch, idx):
    return ' '.join([txt_field.vocab.itos[i] for i in batch.text[0][:,idx].cpu().data.numpy()])

idxtosent(batch,0)

batch.__dict__

val_batch = next(iter(valdl))
val_batch.__dict__
"""


########################

# define class
class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

VOCAB_SIZE = len(txt_field.vocab)
EMBED_DIM = 32
NUN_CLASS = 4
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)

def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry.text for entry in batch]
    offsets = [0] + [len(entry.text) + len(entry.labels) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label



def train_func(sub_train_):

    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=generate_batch)
    for i, (text, offsets, cls) in enumerate(data):
        optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        output = model(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)

def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)

N_EPOCHS = 2
min_valid_loss = float('inf')

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

for epoch in range(N_EPOCHS):

    start_time = time.time()
    train_loss, train_acc = train_func(sub_train_)
    valid_loss, valid_acc = test(sub_valid_)

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')

print('Checking the results of test dataset...')
test_loss, test_acc = test(test_dataset)
print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')

# prediction
ag_news_label = {1 : "World",
                 2 : "Sports",
                 3 : "Business",
                 4 : "Sci/Tec"}

def predict(text, model, vocab, ngrams):
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1

ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."

vocab = train_dataset.get_vocab()
model = model.to("cpu")

print("This is a %s news" %ag_news_label[predict(ex_text_str, model, vocab, 2)])


