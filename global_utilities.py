# -*- coding: utf-8 -*-
"""
Global variables, libraries are defined in this file
"""
from libraries import *

debugging_mode = False
savefig_mode = True

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
gpu_ids = ["0"]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

torch.cuda.empty_cache() # clears gpu

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

"""
We define the global variables we need to price American Options and set CPU
threading. Need to replace tensorflow functions
"""
np_float = np.float64
torch_datatype = torch.float32
global_seed = 0
random.seed(global_seed)
torch.manual_seed(global_seed)

num_cpus = mp.cpu_count()
print("Number of CPUs: ", num_cpus)
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print("Number of GPUs: ", num_gpus)

##############################################
# Experiment Setup

###############################################
##
## Debugging only
##
if False:
    option_type = ['call', 'max', 'vanilla']
    #option_type = ['call', 'geometric', 'vanilla']
    if option_type[1] == "max":
        d = 2
    if option_type[1] == "geometric":
        d = 7
    r = 0.01 # risk free rate #for N = 200 r = 0.01
    div = 0.1 # with div is 0.1 for d => 2 starting with max
    qq = torch.tensor([[div]*d], dtype = torch.float32) # dividend
    mu = r - qq
    sig = 0.3 # 0.25 for T = 1, d => 2 starting with max. 0.2 otherwise # for N = 200, sig = 0.3
    sigma = torch.tensor([[sig]*d], dtype = torch.float32) # volatility
    
    # time for pricing
    t0 = 0

    # for expected exposure
    mu_p = mu
    mu_q = r * torch.ones(d)

    if d == 1:
        rho = torch.eye(d,dtype=torch.float32) 
    if d > 1:
        rho = torch.eye(d,dtype=torch.float32) + 0.3 * (
            torch.ones((d,d), dtype=torch.float32) - torch.eye(d, dtype=torch.float32))
    
    rhoL = torch.cholesky(rho).t() # correlation
    
    if option_type[1] == "geometric":
        eff_d = 1
        srs = sigma.mm(rho.mm(sigma.t()))
        sig_eff = 1 / d * math.sqrt((srs.item()))
        sig_ = sig_eff
        qq_ = (qq + 1 / 2 * sigma ** 2).mean() - 1 / 2 * sig_eff ** 2
        mu_eff = r - qq_
        mu_ = mu_eff.item()
        print("Effective drift:", mu_, "and effective std:", sig_)
    
    if option_type[1] == "max":
        eff_d = d
        sig_eff = (sigma@rho).squeeze(0)
        sig_ = sig_eff[0].item()
        mu_eff = (mu - 1 / 2 * sig_eff ** 2).squeeze(0)
        mu_ = mu_eff[0].item()
        print("Effective drift:", mu_, "and effective std:", sig_)
    
    K = 1.0
    x0 = 0
    x0 = 0.9 * K
    #x0 = K
    #x0 = 1.1 * K
    X0 = torch.tensor([[x0] * eff_d], dtype=torch.float32)
    if eff_d > 1:
        X0 = X0[0]
    #X0 = np.array([[0.9 * K],
    #               [K] ,
    #               [1.1 * K]])
    #X0 = torch.tensor([[K],
    #                [1.1 * K]], dtype = torch.float32)
    S0 = X0
    T = 1
    N = 100
###############################################
##
## Debugging only 2d
##
if False:
    option_type = ['call', 'max', 'vanilla']
    d = 2
    r = 0.05 # risk free rate
    div = 0.1
    qq = torch.tensor([[div]*d], dtype = torch.float32) # dividend
    mu = r - qq
    sig = 0.2
    sigma = torch.tensor([[sig]*d], dtype = torch.float32) # volatility
    rho = torch.eye(d,dtype=torch.float32) + 0.3 * (
        torch.ones((d,d), dtype=torch.float32) - torch.eye(d, dtype=torch.float32))
    rhoL = torch.cholesky(rho).t() # correlation
    sig_eff = (sigma@rhoL).squeeze(0)
    K = 1.0
    #x0 = 0
    x0 = 0.9 * K
    #x0 = K
    #x0 = 1.1 * K
    X0 = torch.tensor([[x0] * d], dtype=torch.float32)
    #X0 = np.array([[0.9 * K] * d,
    #               [K] * d ,
    #               [1.1 * K] * d])
    #X0 = torch.tensor([[K],
    #                [1.1 * K]], dtype = torch.float32)
    
    T = 1.0
    N = 100
#############################################################
##
## Debugging only 200 d
##
if False:
    option_type = ['call', 'geometric', 'vanilla']
    d = 200
    r = 0
    div = 0.02
    qq = torch.tensor([[div] * d], dtype=torch.float32)
    mu = r - qq
    sig = 0.25
    sigma = torch.tensor([[sig] * d], dtype=torch.float32)
    rho = torch.eye(d,dtype=torch.float32) + 0.75 * (
        torch.ones((d,d), dtype=torch.float32) - torch.eye(d, dtype=torch.float32))
    rhoL = torch.cholesky(rho).t() # correlation
    K = 1.0
    # x0 = 0.9 * K
    x0 = K
    # x0 = 1.1 * K
    X0 = torch.tensor([[x0] * d], dtype=torch.float32)
    # x0 = 0
    # X0 = np.array([[0.9 * K] * d,
    #                [K] * d,
    #                [1.1 * K] * d], dtype=np_floattype)
    T = 1.0
    N = 5

## Broadie, Mark, and Jerome Detemple. 1d
## "American option valuation: new bounds, approximations, and a comparison of existing methods."
## The Review of Financial Studies 9.4 (1996): 1211-1250.
if False:
    option_type = ['call', 'max', 'vanilla']
    # option_type = ['call', 'max', 'digital']
    d = 1
    r = 0.03
    div = 0.07
    qq = torch.tensor([[div] * d], dtype=torch.float32)
    mu = r - qq
    sig = 0.2
    sigma = torch.tensor([[sig] * d], dtype=torch.float32)
    rho = torch.eye(d, dtype=torch.float32)
    rhoL = torch.cholesky(rho).t() # correlation
    K = 1.0
    x0 = 0.9 * K
    #x0 = K
    #x0 = 1.1 * K
    X0 = torch.tensor([[x0] * d], dtype=torch.float32)
    # x0 = 0
    # X0 = np.array([[0.9 * K] * d,
    #                [K] * d,
    #                [1.1 * K] * d], dtype=np_floattype)
    T = 0.5
    N = 100

## Kovalov, Pavlo, Vadim Linetsky, and Michael Marcozzi. 3d
## "Pricing multi-asset American options: A finite element method-of-lines with smooth penalty."
## Journal of Scientific Computing 33.3 (2007): 209-237.
if False:
    # option_type = ['put', 'geometric', 'vanilla']
    option_type = ['put', 'arithmic', 'vanilla']
    # d = 2
    d = 3
    # d = 6
    r = 0.03
    div = 0
    qq = torch.tensor([[div] * d], dtype=torch.float32)
    mu = r - qq
    sig = 0.2
    sigma = torch.tensor([[sig] * d], dtype=torch.float32)
    rho = torch.eye(d,dtype=torch.float32) + 0.5 * (
        torch.ones((d,d), dtype=torch.float32) - torch.eye(d, dtype=torch.float32))
    rhoL = torch.cholesky(rho).t() # correlation

    K = 1.0
    # x0 = 0.9 * K
    x0 = K
    # x0 = 1.1 * K
    X0 = torch.tensor([[x0] * d], dtype=torch.float32)
    # x0 = 0
    # X0 = np.array([[0.9 * K] * d,
    #                [K] * d,
    #                [1.1 * K] * d], dtype=np_floattype)
    T = 0.25
    N = 100

## Broadie, Mark, and Paul Glasserman. 2d
## "Pricing American-style securities using simulation."
## Journal of economic dynamics and control 21.8-9 (1997): 1323-1352.
if False:
    # time for pricing
    t0 = int(0)

    option_type = ['call', 'max', 'vanilla']
    d = 2
    r = 0.05
    div = 0.1
    qq = torch.tensor([[div] * d], dtype=torch.float32)
    mu = r - qq
    sig = 0.2
    sigma = torch.tensor([[sig] * d], dtype=torch.float32)
    rho = torch.eye(d, dtype=torch.float32) + 0.3 * (
        torch.ones((d, d), dtype=torch.float32) - torch.eye(d, dtype=torch.float32))
    rhoL = torch.cholesky(rho).t() # correlation

    if option_type[1] == "max":
        eff_d = d
        sig_eff = 1 / eff_d * math.sqrt(torch.matmul(sigma,torch.matmul(rho,sigma.transpose(0,1))))
        qq_ = torch.mean(qq + 1 / 2 * sigma ** 2) - 1 / 2 * sig_eff ** 2
        mu_eff = (r - qq_).item()
        mu_,sig_ = mu_eff, sig_eff
        print("Effective drift:", mu_, "and effective std:", sig_)

    K = 1.0
    #x0 = 0.8 * K
    x0 = 0.9 * K
    x0 = K
    x0 = 1.1 * K
    #x0 = 1.2*K
    X0 = torch.tensor([[x0] * eff_d], dtype=torch.float32)
    # x0 = 0
    # X0 = np.array([[0.9 * K] * d,
    #                [K] * d,
    #                [1.1 * K] * d], dtype=np_floattype)
    S0 = X0[0]
    T = 1.0
    N = 288

## Firth, Neil Powell.
## High dimensional American options.
## Diss. University of Oxford, 2005.
if False:
    option_type = ['call', 'max', 'vanilla']
    d = 5
    r = 0.05
    div = 0.1
    qq = torch.tensor([[div] * d], dtype=torch.float32)
    mu = r - qq
    sig = 0.2
    sigma = torch.tensor([[sig] * d], dtype=torch.float32)
    rho = torch.eye(d, dtype=torch.float32)
    rhoL = torch.cholesky(rho).t() # correlation
    if option_type[1] == "max":
        eff_d = d
        sig_eff = 1 / eff_d * math.sqrt(torch.matmul(sigma,torch.matmul(rho,sigma.transpose(0,1))))
        qq_ = torch.mean(qq + 1 / 2 * sigma ** 2) - 1 / 2 * sig_eff ** 2
        mu_eff = (r - qq_).item()
        mu_,sig_ = mu_eff, sig_eff
        print("Effective drift:", mu_, "effective std:", sig_)
    
    K = 1.0
    x0 = 0.9 * K
    x0 = K
    x0 = 1.1 * K
    X0 = torch.tensor([[x0] * d], dtype=torch.float32)
    # x0 = 0
    # X0 = np.array([[0.9 * K] * d,
    #                [K] * d,
    #                [1.1 * K] * d], dtype=np_floattype)
    S0 = X0[0]
    T = 3.0
    N = 100
    Nmax = N

## Sirignano, Justin, and Konstantinos Spiliopoulos. 10d
## "DGM: A deep learning algorithm for solving partial differential equations."
## Journal of Computational Physics 375 (2018): 1339-1364.
if True:
    option_type = ['call', 'geometric', 'vanilla']
    d = 5
    r = 0
    div = 0.02
    qq = torch.tensor([[div] * d], dtype=torch.float32)
    mu = r - qq
    sig = 0.25
    sigma = torch.tensor([[sig] * d], dtype=torch.float32)
    rho = torch.eye(d, dtype=torch.float32) + 0.75 * (
        torch.ones((d, d), dtype=torch.float32) - torch.eye(d, dtype=torch.float32))
    rhoL = torch.cholesky(rho).t() # correlation

    if option_type[1] == "geometric":
        eff_d = d
        sig_eff = 1 / eff_d * math.sqrt(torch.matmul(sigma,torch.matmul(rho,sigma.transpose(0,1))))
        qq_ = torch.mean(qq + 1 / 2 * sigma ** 2) - 1 / 2 * sig_eff ** 2
        mu_eff = (r - qq_).item()
        mu_,sig_ = mu_eff, sig_eff
        print("Effective drift:", mu_, "effective std:", sig_)

    K = 1.0
    #x0 = 0.9 * K
    x0 = K
    #x0 = 1.1 * K
    X0 = torch.tensor([[x0] * eff_d], dtype=torch.float32)
    # x0 = 0
    # X0 = np.array([[0.9 * K] * d,
    #                [K] * d,
    #                [1.1 * K] * d], dtype=np_floattype)
    S0 = X0[0]
    T = 2.0
    N = 100

## Kristoff, Andersson, and Oosterlee, Cornelis. 5d & 30d
## "A deep learning approach for computations of exposure profile for high dim Bermudan option."
if False:
    # time for pricing
    t0 = int(0)

    option_type = ['call', 'max', 'vanilla']
    d = 30
    r = 0.05
    div = 0.1
    qq = torch.tensor([[div] * d], dtype=torch.float32)
    mu = r - qq
    sig = 0.2
    sigma = torch.tensor([[sig] * d], dtype=torch.float32)
    rho = torch.eye(d, dtype=torch.float32) + 0 * (
        torch.ones((d, d), dtype=torch.float32) - torch.eye(d, dtype=torch.float32))
    rhoL = torch.cholesky(rho).t() # correlation

    if option_type[1] == "max":
        eff_d = d
        sig_eff = 1 / eff_d * math.sqrt(torch.matmul(sigma,torch.matmul(rho,sigma.transpose(0,1))))
        qq_ = torch.mean(qq + 1 / 2 * sigma ** 2) - 1 / 2 * sig_eff ** 2
        mu_eff = (r - qq_).item()
        mu_,sig_ = mu_eff, sig_eff
        print("Effective drift:", mu_, "and effective std:", sig_)

    K = 1.0
    x0 = 0.9 * K
    x0 = K
    x0 = 1.1 * K
    X0 = torch.tensor([[x0] * eff_d], dtype=torch.float32)
    # x0 = 0
    # X0 = np.array([[0.9 * K] * d,
    #                [K] * d,
    #                [1.1 * K] * d], dtype=np_floattype)
    S0 = X0[0]
    T = 3.0
    N = 100

#############################################
scale = 100

# simulating processe
dt = T/N # step size
simulation_size = 720000 # simulation size
sharpness = 2/dt
payoff_tol = 1e-8
n_training_step = N
seq_len = N
##############################################
# variables used in rnn
requires_grad = True
dropout = 0

batch_size = int(simulation_size/300)
batch_size_test = int(simulation_size/100)
num_epochs = 1

input_size = 2
output_size = 1

num_layer = 5
hidden_size = 5
embd_size = input_size