

from torch.serialization import load
import torch 
from optimus_prime import TransformerWrapper, Decoder, AutoregressiveWrapper

#training
import random
import tqdm
import gzip
import numpy as np
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import os
# from torch.utils.tensorboard import SummaryWriter
# from torchmetrics import MetricCollection, Accuracy


# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
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

model = TransformerWrapper(
    num_tokens=20000,
    max_seq_len=200000,
    use_abs_pos_emb = False,
    attn_layers = Decoder(
        dim=512,
        depth=6,
        heads=8,
        alibi_pos_bias=True,
        alibi_num_heads=4,
        rotary_xpos=True,
        attn_flash = True,
        deepnorm=True,
        # dynamic_pos_bias=True,
        # dynamic_pos_bias_log_distance=False,
        shift_tokens=1,
        # rel_pos_bias=True
    )
)


model = AutoregressiveWrapper(model)
model.cuda()

with gzip.open('./enwik8.gz') as file:
  data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
  train_x, valid_x = np.split(data, [int(90e6)])
  data_train, data_val = torch.from_numpy(train_x), torch.from_numpy(valid_x) #.cuda()??

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
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE, drop_last = True))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE, drop_last = True))

# optimizer

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training

# #init tensorboard 
# writer = SummaryWriter(log_dir="./log")

# #define metrics
# metrics = MetricCollection({'accuracy': Accuracy(num_classes=num_classes, task='classification')})

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(next(train_loader))
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    print(f'training loss: {loss.item()}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()


    if i % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                loss = model(next(val_loader))
                print(f'validation loss: {loss.item()}')

                # # Calculate validation metrics
                # val_metrics = MetricCollection({'val_accuracy': Accuracy()})
                # val_metrics(loss, model(next(val_loader)).argmax(dim=-1))

                # # Add validation metrics to the SummaryWriter
                # writer.add_scalar('Validation/Accuracy', val_metrics['val_accuracy'].compute(), global_step=i)

    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:-1]
        prime = decode_tokens(inp)
        print(f'%s \n\n %s', (prime, '*' * 100))

        sample = model.generate(inp, GENERATE_LENGTH)
        output_str = decode_tokens(sample)
        print(output_str)

    # Save the model every save_every iterations
    if i % SAVE_EVERY == 0:
        # Specify the directory and filename to save the model
        save_dir = './saved_models/'
        save_filename = 'model_checkpoint.pt'

        # Create the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Save the model checkpoint
        torch.save(model.state_dict(), os.path.join(save_dir, save_filename))
        print(f"Model saved at iteration {i}")

#     # Add training metrics to the SummaryWriter
#     writer.add_scalar('Training/Accuracy', metrics['accuracy'].compute(), global_step=i)

#     # Close the SummaryWriter
# writer.close()