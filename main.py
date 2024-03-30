""" microGPT """
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'[-] Device: {device}')

with open('input-dataset.txt', 'r', encoding='utf-8') as file:
    text = file.read()

print(f"[-] Length of dataset in characters: \n{len(text)} \n")
print(f"[-] File size: \n{round(os.path.getsize('input-dataset.txt') / (1024 * 1024), 2)} MB \n" )

# have a look at the first thousand characters:

print(text[:1000] + '\n')

# Extract all the unique characters that occur in this text:

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"[-] Unique characters: ({vocab_size})\n {''.join(chars)} \n")

# TOKENIZER (character level)

""" create a mapping from characters to integers """

string_to_integer = { ch:i for i, ch in enumerate(chars) }
integer_to_string = { i:ch for i, ch in enumerate(chars) }

#ENCODER: take a string, output a list of integers
encode = lambda s: [ string_to_integer[ch] for ch in s ]
#DECODER: take a list of integers, output a string
decode = lambda l: [ ''.join([integer_to_string[i] for i in l]) ]

print(f"[-] Test encoder: hello world! = {encode("hello world!")}")
print(f"[-] Test encoder: Hello World! = {encode("Hello World!")}")

print(f"[-] Test decoder: hellow world! = {decode(encode("hello world!"))}")
print(f"[-] Test decoder: hellow world! = {decode(encode("Hello World!"))}")


# TOKENIZE DATASET


data = torch.tensor(encode(text), dtype=torch.long)

print(data.shape, data.dtype)
print(data[:1000])
# --------------------------

# Set train and validation data:

n = int(0.9*len(data)) # first 90% for training rest for validation
train_data = data[:n]
val_data = data[n:]

block_size = 8 # maximum context length
train_data[:block_size+1]


torch.manual_seed(1337)
batch_size = 4 # independet parallel sequences

# DATA CHUNK LOADER

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device) , y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    eval_iters = 1000
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

xb, yb, = get_batch('train')
print('--------------------------')
print(f'[-] Input batch shape: {xb.shape} \n {xb}')
print(f'[-] Target batch shape: {yb.shape} \n {yb}')
print('--------------------------')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f'[-] Context: {context.tolist()} -> Target: {target}')

# BIGRAM LANGUAGE MODEL
        
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx) # (Batch, Time, Channel)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F. cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)
            #focus only in the last time step
            logits = logits[:, -1, :] # Becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            sample = torch.multinomial(probs, num_samples=1)
            # append to the sequence
            idx = torch.cat((idx, sample), dim=1)
        return idx


m = BigramLanguageModel(vocab_size)
model = m.to(device)

logits, loss = model(xb, yb)
# DATASET SIZE
print(logits.shape, loss)


print(f'[-] pre-training generation: {decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist())}')



# TRAINING

#create a pytorch optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

batch_size = 32
max_iters = 10000
eval_interval = 300
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'[-] Step: {iter}, Train loss: {losses["train"]}, Val loss: {losses["val"]}')

    #sample batch of data
    xb, yb = get_batch('train')

    #evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

print(f'[-] AFTER training generation: {decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=300)[0].tolist())}')



