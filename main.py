# -*- coding: utf-8 -*-
from libraries import *
from rnn_function import *
from global_utilities import *
from simulation import *

S = load_price_data()

print(f"Simulation size: {S.shape[0]}")
Value = []
Delta = []
Value0 = []
Delta0 = []
Continuation_Price = []
rnn = []
optimizer = []

rnn_p = price_rnn()
rnn_d = delta_rnn()

if torch.cuda.is_available():
    rnn_p = rnn_p.cuda()
    rnn_d = rnn_d.cuda()
    
optimizer_p = torch.optim.Adam(rnn_p.parameters())
optimizer_d = torch.optim.Adam(rnn_d.parameters())

loader = DataLoader(TensorDataset(S.clone()), batch_size=batch_size, shuffle=True)
rnn.append((rnn_p, rnn_d))
optimizer.append((optimizer_p, optimizer_d))

# training
for e in tqdm(range(num_epochs)):
    print(f"\nTraining for Epoch: {e+1} out of {num_epochs}\n")
    train_rnn(rnn, optimizer, loader)

loader_test = DataLoader(TensorDataset(S), batch_size=batch_size_test, shuffle=False)
with torch.no_grad():
    Value, Delta, Cont_price, time_price = predict_rnn(rnn, loader_test) # outputs are from t = 0, ...., T

V0 = Value[0].mean(0)
D0 = Delta[0].mean(0)

for i in range(eff_d):
    print("Initial price:", format(S0[i], '.1f'), "for asset", i + 1)
    print("The price at t = 0 for Maturity T =", format(T, '.1f'), ":", format(V0[i], '.6f'))
    print("Std dev of Price: ", np.std(np.array(V0)))
    print("Delta at t = 0:", format(D0[i],'.6f'))
    print("Std dev of Delta: ", np.std(np.array(D0)), '\n')
