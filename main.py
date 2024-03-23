""" microGPT """
import os

with open('input-dataset.txt', 'r', encoding='utf-8') as file:
    text = file.read()

print(f"[-] Length of dataset in characters: \n{len(text)} \n")
print(f"[-] File size: \n{round(os.path.getsize('input-dataset.txt') / (1024 * 1024), 2)} MB \n" )

# have a look at the first thousand characters:

print(text[:1000] + '\n')

# Extract all the unique characters that occur in this text:

chars = sorted(list(set(text)))
print(f"[-] Unique characters: ({len(chars)})\n {''.join(chars)} \n")

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
import torch

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

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

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





