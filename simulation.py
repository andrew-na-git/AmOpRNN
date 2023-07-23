from libraries import *
from payoff import *
from rnn_function import *
from global_utilities import *

###############################
# index 0 is time, index 1 is state
# the input dimensions to RNN requires dim of (seq_len, batch_size, input_size)
# hidden and cell state requires dim of (num_layers, batch_size, hidden_size)

def train_rnn(rnn, optimizer, loader):

    price_rnn = rnn[0][0].train()
    delta_rnn = rnn[0][1].train()

    price_opt = optimizer[0][0]
    delta_opt = optimizer[0][1]

    print("\nOptimizing\n")
    t0 = time.time()

    for data in tqdm(loader):

        s = data[0].transpose(0,1) # .transpose(0,1) -> shape (seq_len, batch_size, d)
        s = Variable(torch.flip(s, dims=[0])) # starts from T to 0
        
        if option_type[1] == "geometric":
            x, dxds = Geometric(s)
            x = x.repeat((1,1,eff_d))

        if option_type[1] == "max":
            x, dxds = Max(s) # shape (seq_len, batch_size, 1)
            x = x.repeat((1,1,eff_d))
            # dxds = (s > x - 1e-6)

        if torch.cuda.is_available():
            dxds = dxds.cuda()
            s = s.cuda()
            x = x.cuda()

        # shape (seq_len, batch_size, eff_d)
        if option_type[0] == "call":
            u = F.softplus(call(x,K),beta=sharpness)
            z = torch.sigmoid(sharpness*call(x,K))*dxds

        if option_type[0] == "put":
            u = F.softplus(put(x,K),beta=sharpness)
            z = torch.sigmoid(sharpness*put(x,K))*dxds

        # initialize variables
        p = torch.empty_like(z)
        dpds = torch.empty_like(z)
        p_star = torch.empty_like(z)
        d_star = torch.empty_like(z)
        disc_v = torch.empty_like(z[1:])
        disc_d = torch.empty_like(z[1:])

        # construct values at T
        p_star[0] = u[0]
        d_star[0] = z[0]

        disc_v[0] = u[0]
        disc_d[0] = z[0]

        p[0] = u[0]
        dpds[0] = z[0]
        
        for t in range(seq_len):
            q = t+1

            if q == 1:
                p_star[q] = math.exp(-r*dt)*u[t]
                d_star[q] = math.exp(-r*dt)*z[t]*(s[t]/s[q])
            if q > 1:
                _,stop_idx = torch.max(u[:q+1].mean(1),0)
                for k in range(eff_d):
                    delta_tau = q - stop_idx[k]
                    p_star[q,:,k] = math.exp(-r*delta_tau*dt)*u[int(stop_idx[k]),:,k]
                    d_star[q,:,k] = math.exp(-r*delta_tau*dt)*(z[int(stop_idx[k]),:,k]*(s[int(stop_idx[k]),:,k]/s[q,:,k]))
            
                disc_v[t] = math.exp(-r*(q-1)*dt)*p_star[1]
                disc_d[t] = math.exp(-r*(q-1)*dt)*d_star[1]*(s[1]/s[q])

        # print(p_star.mean(1), d_star.mean(1))
        # input()
        
        for k in range(eff_d):
            in_p = torch.cat((s[1:,:,k].unsqueeze(-1),x[1:,:,k].unsqueeze(-1)),dim=-1) # (seq_len, batch_size, num_features/inputs)
            hp = price_rnn.init_hidden(u[1,:,k].unsqueeze(-1))
            p[1:,:,k] = price_rnn(in_p,hp,disc_v[:,:,k])
            
            in_d = torch.cat((s[1:,:,k].unsqueeze(-1),x[1:,:,k].unsqueeze(-1)),dim=-1) # (seq_len, batch_size, num_features/inputs)
            hd = delta_rnn.init_hidden(z[1,:,k].unsqueeze(-1))
            dpds[1:,:,k] = delta_rnn(in_d,hd,disc_d[:,:,k])

        print(p[-1].mean(0), dpds[-1].mean(0))
        price_opt.zero_grad()
        delta_opt.zero_grad()
        loss_p, loss_d = loss_sde(p,p_star,dpds,d_star)
        loss = loss_p + loss_d
        loss.backward()
        price_opt.step()
        delta_opt.step()
        print(f"\nThe batch loss: {loss.clone().detach().cpu().numpy()}\n")
            
    tf = time.time() - t0
    #print(torch.cuda.memory_summary())
    print("\nTraining took:", tf, "\n")

def predict_rnn(rnn, loader):

    price_rnn = rnn[0][0].eval()
    delta_rnn = rnn[0][1].eval()
    
    Value = []
    Delta = []
    Exp = []

    print("pricing asset")
    t0 = time.time()

    for data in tqdm(loader):
        s = data[0].transpose(0,1) # .transpose(0,1) -> shape (seq_len, batch_size, d)
        s = torch.flip(s, dims=[0]) # this inverts the timesteps
        
        if option_type[1] == "geometric":
            x, dxds = Geometric(s)
            x = x.repeat((1,1,eff_d))

        if option_type[1] == "max":
            x, dxds = Max(s) # shape (seq_len, batch_size, 1)
            x = x.repeat((1,1,eff_d))
            # dxds = (s > x - 1e-6)

        if torch.cuda.is_available():
            dxds = dxds.cuda()
            s = s.cuda()
            x = x.cuda()
        
        # shape (seq_len, batch_size, 1)
        if option_type[0] == "call":
            u = F.softplus(call(x,K),beta=sharpness)
            z = torch.sigmoid(sharpness*call(x,K))*dxds

        if option_type[0] == "put":
            u = F.softplus(put(x,K),beta=sharpness)
            z = torch.sigmoid(sharpness*put(x,K))*dxds

        # initialize variables
        value = torch.empty_like(z)
        delta = torch.empty_like(z)
        p = torch.empty_like(z)
        dpds = torch.empty_like(z)
        cp = torch.empty_like(z)
        cp_delta = torch.empty_like(z)
        disc_v = torch.empty_like(z[1:])
        disc_d = torch.empty_like(z[1:])

        # construct values at T
        value[0] = u[0]
        delta[0] = z[0]

        cp[0] = u[0]
        cp_delta[0] = z[0]

        disc_v[0] = u[0]
        disc_d[0] = z[0]

        p[0] = u[0]
        dpds[0] = z[0]

        for t in range(seq_len):
            q = t+1

            if q == 1:
                cp[q] = math.exp(-r*dt)*u[t]
                cp_delta[q] = math.exp(-r*dt)*z[t]*(s[t]/s[q])
            if q > 1:
                #_,stop_idx = torch.max(u[:q+1].mean(1),0)
                #for k in range(eff_d):
                    #delta_tau = q - stop_idx[k]
                    #cp[q,:,k] = math.exp(-r*delta_tau*dt)*u[int(stop_idx[k]),:,k]
                    #cp_delta[q,:,k] = math.exp(-r*delta_tau*dt)*(z[int(stop_idx[k]),:,k]*(s[int(stop_idx[k]),:,k]/s[q,:,k]))
                disc_v[t] = math.exp(-r*(q-1)*dt)*cp[1]
                disc_d[t] = math.exp(-r*(q-1)*dt)*cp_delta[1]*(s[1]/s[q])

        for k in range(eff_d):
            in_p = torch.cat((s[1:,:,k].unsqueeze(-1),x[1:,:,k].unsqueeze(-1)), dim=-1) # (seq_len, batch_size, num_features/inputs)
            hp = price_rnn.init_hidden(u[1,:,k].unsqueeze(-1))
            p[1:,:,k] = price_rnn(in_p,hp,disc_v[:,:,k])

            in_d = torch.cat((s[1:,:,k].unsqueeze(-1),x[1:,:,k].unsqueeze(-1)), dim=-1)# (seq_len, batch_size, num_features/inputs)
            hd = delta_rnn.init_hidden(z[1,:,k].unsqueeze(-1))
            dpds[1:,:,k] = delta_rnn(in_d,hd,disc_d[:,:,k])
        
        idx = (u[1:] > p[1:])
        value[1:] = u[1:]*(idx) + disc_v[1:]*(~idx)
        delta[1:] = z[1:]*(idx) + disc_d[1:]*(~idx)
        
        print(value[-1].mean(0),delta[-1].mean(0))

        Value.append(value.clone().detach().cpu())
        Delta.append(delta.clone().detach().cpu())
        Exp.append(p.clone().detach().cpu())

    tf = time.time() - t0
    print("\nPricing took:", tf, "seconds\n")

    # shape (seq_len, batch_size, eff_d)
    Value = torch.flip(torch.cat(Value, dim=1), dims=[0])
    Delta = torch.flip(torch.cat(Delta, dim=1), dims=[0])
    Exp = torch.flip(torch.cat(Exp, dim=1), dims=[0])

    return Value, Delta, Exp, tf
