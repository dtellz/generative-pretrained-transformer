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

