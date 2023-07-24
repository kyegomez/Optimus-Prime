
import os
import gzip
import tqdm
import torch
import random
import numpy as np
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from optimus_prime import TransformerWrapper, Decoder, AutoregressiveWrapper, AndromedaEmbedding


# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 2
LEARNING_RATE = 1e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
GENERATE_LENGTH = 1024
SEQ_LEN = 1024
SAVE_EVERY=500


# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))


# Model
model = TransformerWrapper(
    num_tokens=20000,
    max_seq_len=8192,
    use_abs_pos_emb=False,
    embedding_provider=AndromedaEmbedding(),
    attn_layers=Decoder(
        dim=512,
        depth=6,
        heads=8,
        use_abs_pos_emb=False, 
        alibi_pos_bias=True, 
        alibi_num_heads=4, 
        rotary_xpos=True,
        attn_flash=True, 
        # shift_tokens=1, 
        attn_one_kv_head=True, 
        qk_norm=True, 
        attn_qk_norm=True, 
        attn_qk_norm_dim_scale=True, 
    )
)

model = AutoregressiveWrapper(model)
model.cuda()

# Dataset
with gzip.open('./enwik8.gz') as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    train_x, valid_x = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(train_x), torch.from_numpy(valid_x)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Optimizer
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# Training
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss, _ = model(next(iter(train_loader)))  # Unpack the tuple here
        loss = loss.mean()  # Ensure the loss is a scalar
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    print(f'training loss: {loss.item()}')
    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    optimizer.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            val_loss, _ = model(next(iter(val_loader)))  # Unpack the tuple here
            val_loss = val_loss.mean()  # Ensure the loss is a scalar 
            print(f'validation loss: {val_loss.item()}')

    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:-1]
        sample = model.generate(inp, GENERATE_LENGTH)
        output_str = decode_tokens(sample)
        print(output_str) #

    if i % SAVE_EVERY == 0:
        torch.save(model.state_dict(), f'./saved_models/model_checkpoint_{i}.pt')
        print(f"Model saved at iteration {i}")