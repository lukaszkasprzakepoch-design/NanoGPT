import numpy as np
import torch
import os


with open("input.txt",'r',encoding = 'utf-8') as f:
    text = f.read()
print(text[:100])

print("lenght",len(text))
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[si] for si in s]    #takes string, outputs tokens
decode = lambda n: ''.join([itos[ni] for ni in n])    #takes tokens, outputs a string

s = "assss"
encoded = encode(s)
decoded = decode(encoded)
print(s)
print(decoded)
assert s == decoded

print("HELLO WORLD")

torch.manual_seed(1337)
batch_size = 4
block_size = 8
data = None

n = int(0.9 * len(data))
train_data = data[:n-6]
val_data = data[n:]

def get_batch(split):
    if split == 'train':
        data = train_data 
    else:
        data = val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([train_data[x:x+block_size] for x in ix])
    y = torch.stack([train_data[x+1:x+block_size+1] for x in ix])
    return x,y

xb,yb = get_batch('train')
print('inputs:',xb.shape,xb)
print(xb)
print(yb)
