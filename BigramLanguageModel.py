import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import argparse
from datetime import datetime
import wandb  # type: ignore

_WANDB_AVAILABLE = False

batch_size = 4
block_size = 8
max_tokens = 1000
max_iters = 10000
eval_iters = 200
eval_interval = 1000
learning_rate = 1e-3
n_embd = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device)

def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x,y

torch.manual_seed(1337)

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb,yb = get_batch(train_data if split == 'train' else val_data)
            logits,loss = m(xb,yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)

    def forward(self,idx,targets=None):
        B,T = idx.shape

        #Let's add the average of all previous tokens as a simple context
        wei = torch.tril(torch.ones(T,T)) # (T,T)
        wei = wei / wei.sum(dim=1,keepdim=True)
        wei = wei.to(idx.device)
        #print("wei shape:", wei.shape)

        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        #print("tok_emb shape:", tok_emb.shape)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        #print("pos_emb shape:", pos_emb.shape)

        x = tok_emb + pos_emb # (B,T,C)
        #print("x shape after adding token and position embeddings:", x.shape)
        x = x.transpose(1, 2)   # (4, 32, 8)
        x = x @ wei.T           # (4, 32, 8)
        x = x.transpose(1, 2)   # (4, 8, 32)
        logits = self.lm_head(x) # (B,T,C)
        
        #print("logits shape:",logits.shape)
        #print("logits:",logits)
        #print("idx",idx)
        #print("targets",targets)
        if targets is None:
            loss = None
        else: 
            B,T,C = logits.shape  #bacth size, block size, vocab size
            logits = logits.view(B*T,C) #reshape the logits to be a 2D tensor of shape (B*T, C)
            #print("logits shape after view:",logits.shape)
            #print("logits after view:",logits)
            targets = targets.view(B*T) 
            #print("targets shape after view:",targets.shape)
            #print("targets after view:",targets)
            loss = F.cross_entropy(logits,targets)
            #print("loss:",loss.item())
        
        return logits,loss
    
    def generate(self,idx,max_new_tokens):
        #idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # only feed the last `block_size` tokens so positional indices stay in-range
            idx_cond = idx[:, -block_size:]
            logits,loss = self(idx_cond)
            #print("logits shape in generate:",logits.shape)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", type=str, default="nanogpt")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    args = parser.parse_args()

    with open("input.txt",'r',encoding = 'utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    encode = lambda s: [stoi[si] for si in s]    #takes string, outputs tokens
    decode = lambda n: ''.join([itos[ni] for ni in n])    #takes tokens, outputs a string
    data = torch.tensor(encode(text),dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[:n-6]
    val_data = data[n:]

    m = BigramLanguageModel(vocab_size)
    m = m.to(device)

    use_wandb = bool(args.wandb) and _WANDB_AVAILABLE
    if bool(args.wandb) and not _WANDB_AVAILABLE:
        print("wandb not installed; continuing without logging. Install with: pip install wandb")

    if use_wandb:
        run_name = args.wandb_run_name or f"bigram-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "batch_size": batch_size,
                "block_size": block_size,
                "max_iters": max_iters,
                "eval_iters": eval_iters,
                "eval_interval": eval_interval,
                "learning_rate": learning_rate,
                "n_embd": n_embd,
                "device": device,
                "vocab_size": vocab_size,
            },
        )
    
    # xb,yb = get_batch(train_data)
    # #lets now make a simple context (average all previous tokens)
    wei = torch.tril(torch.ones(block_size,block_size)) # (B,T,T)
    wei = wei / wei.sum(dim=1,keepdim=True) # (B,T,T)
    wei = wei.to(device)
    # print("wei shape:",wei.shape)
    # print(wei)
    # print("xb shape:",xb.shape)
    # print(xb)
    # print((xb @ wei.T)) # (B,T,C)

    optimizer = torch.optim.AdamW(m.parameters(),lr=1e-3)

    for steps in tqdm.tqdm(range(max_iters)):
        xb,yb = get_batch(train_data)
        #print("xb shape:", xb.shape)
        #print("yb shape:", yb.shape)
        logits,loss = m(xb,yb)
        optimizer.zero_grad(set_to_none=True)
        #print("model parameters shape before backward:", [p.shape for p in m.parameters()])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if use_wandb:
            wandb.log({"train/loss": loss.item(), "iter": steps}, step=steps)

        if steps % eval_interval == 0:
            eval_loss = estimate_loss()
            print(steps, eval_loss)
            if use_wandb:
                wandb.log(
                    {"eval/train_loss": float(eval_loss["train"]), "eval/val_loss": float(eval_loss["val"])},
                    step=steps,
                )
    print(torch.zeros((1,1)))
    print(decode(m.generate(idx=torch.zeros((1,1),dtype=torch.long,device=device),max_new_tokens=max_tokens)[0].tolist()))

    if use_wandb:
        wandb.finish()

