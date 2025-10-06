import torch
from collections import defaultdict

def mae(pred, target): return torch.mean(torch.abs(pred-target), dim=0)  # per-dim

class AvgMeter:
    def __init__(self): self.n=0; self.sum=0.0
    def update(self, val, n=1): self.sum+=float(val)*n; self.n+=n
    @property
    def avg(self): return self.sum/max(1,self.n)

class DictMeter:
    def __init__(self): self.m=defaultdict(AvgMeter)
    def update(self, d, n=1):
        for k,v in d.items(): self.m[k].update(v, n)
    def avg(self): return {k:v.avg for k,v in self.m.items()}
