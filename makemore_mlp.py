#%%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
# %%
names = open("data/names.txt", "r").read().splitlines()
names[:5]
# %%
chars = sorted(set("".join(names)))
stoi = {chars[i]:i+1 for i in range(len(chars))}
stoi["."] = 0
itos = {c:i for i,c in stoi.items()}
def decode(ixs):
    if ixs.numel() == 1:
        return itos[int(ixs)]
    return "".join([itos[int(ix)] for ix in ixs])
# %%
block_size = 3
X, Y = [], []
for w in names:
    context = [0] * block_size
    for ch in w + ".":
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]
X, Y = torch.tensor(X), torch.tensor(Y)

# %%
lr = 0.1
batch_size = 32
gen = torch.Generator().manual_seed(42)
C = torch.randn(27, 2, generator=gen)
W1 = torch.randn(6, 100, generator=gen)
b1 = torch.zeros(100)
W2 = torch.randn(100, 27, generator=gen)
b2 = torch.zeros(27)
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True
#%%
lr = 0.01
for i in range(10000):
    minibatch_idx = torch.randint(0, X.shape[0], (batch_size,))
    X_batch, Y_batch = X[minibatch_idx], Y[minibatch_idx]
    emb = C[X_batch].view(-1, 6)
    h = emb @ W1 + b1
    h = torch.tanh(h)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y_batch)
    for p in parameters:
        p.grad = None
    loss.backward()
    for p in parameters:
        p.data += -lr * p.grad
    #print(loss.item())
# %%
emb = C[X].view(-1, 6)
h = emb @ W1 + b1
h = torch.tanh(h)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Y)        
loss
# %%
