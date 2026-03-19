import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([train_data[x:x+block_size] for x in ix])
    y = torch.stack([train_data[x+1:x+block_size+1] for x in ix])
    return x,y

torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self,idx,targets=None):
        logits = self.token_embedding_table(idx)
        #print("logits shape:",logits.shape)
        #print("logits:",logits)
        #print("idx",idx)
        #print("targets",targets)
        if targets is None:
            loss = None
        else: 
            B,T,C = logits.shape  #bacth size, block size, vocab size
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)

        return logits,loss
    
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            logits,loss = self(idx)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx
    
if __name__ == "__main__":
    with open("input.txt",'r',encoding = 'utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    encode = lambda s: [stoi[si] for si in s]    #takes string, outputs tokens
    decode = lambda n: ''.join([itos[ni] for ni in n])    #takes tokens, outputs a string

    batch_size = 4
    block_size = 8
    data = torch.tensor(encode(text),dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[:n-6]
    val_data = data[n:]

    m = BigramLanguageModel(vocab_size)

    xb,yb = get_batch(train_data)
    logits,loss = m(xb,yb)
    print("logits shape:",logits.shape)
    print("loss:",loss)

    print(decode(m.generate(idx=torch.zeros((1,1),dtype=torch.long),max_new_tokens=100)[0].tolist()))

    optimizer = torch.optim.AdamW(m.parameters(),lr=1e-3)

    for steps in tqdm.tqdm(range(10000)):
        xb,yb = get_batch(train_data)
        logits,loss = m(xb,yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if steps % 1000 == 0:
            print(steps,loss.item())

    print(decode(m.generate(idx=torch.zeros((1,1),dtype=torch.long),max_new_tokens=100)[0].tolist()))

