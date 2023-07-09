from libraries import *
from global_utilities import *

def call(S,K):
    return S-K

def put(S,K):
    return K-S

def ContinueBound(f, h):
    boundary = (f < h) | (f < payoff_tol)
    return boundary

def AmericanOP(f, h):
    payoff = torch.max(f, h)
    return payoff

def EuropeanOP(f):
    return f

def Geometric(s):
    x = torch.exp(torch.mean(torch.log(torch.abs(s)),dim=-1,keepdim=True))
    dxds = 1/d * x * (1/s)
    return x, dxds

def Arithmetic(s):
    x = x.numpy()
    x = trimmed_mean(s,axis=2)
    x = torch.tensor(x)
    dxds = (1/d) * torch.ones_like(x).repeat((1,1,eff_d))
    return x, dxds

def Max(s):
    x = (1/sharpness)*torch.logsumexp(sharpness*s,dim=-1,keepdim=True)
    dxds = torch.softmax(sharpness*s, dim=-1)
    return x, dxds