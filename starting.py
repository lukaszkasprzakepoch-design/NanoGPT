import numpy as np
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