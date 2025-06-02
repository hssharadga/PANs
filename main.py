
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 20:50:58 2025

@author: Hussein Sharadga
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import copy
import random
import gurobipy as gp
from gurobipy import GRB
import random
# Time Count
import timeit
start = timeit.default_timer()


# Calculate Frobenius norms
def frobenius_norm(matrix1, matrix2):
    return np.linalg.norm(matrix1 - matrix2)





# A: Import School Results
school=pd.read_csv('school.csv')
school_test=school[329088-96*365+96*4-1:329088-2]# Load profile for one year, starts on January 1, 2010 at 12:00 AM (00:00 in 24 hr style)
school_test=school_test[0:360*24*4] # The data is recorded every 15 minutes, thus 4*24 points for one day and 360 is assumed to be the 12 months  (We know that 12 months is 365 days)
school_test=np.array(school_test)
school_test=np.reshape(school_test,(360*96))
# Averaging: 15 mins to hourly
school_test=np.reshape(school_test,(360*24,4))
school_test=np.mean(school_test,axis=1)         # Hourly demnad for 360 days

# One year for training and one year for test
x=329088-96*365+96*4-1
school_train=school[x-96*365-1:x-2]
school_train=school_train[1*96:-(4)*96+1]
school_train=np.array(school_train)
school_train=np.reshape(school_train,(360*96))
# Averaging: 15 mins to hourly
school_train=np.reshape(school_train,(360*24,4))
school_train=np.mean(school_train,axis=1)   

# Work days only: remove the weekends
school_train_work=[]
school_test_work=[]
for i in range(51):
    school_train_work=np.concatenate((school_train_work,school_train[24*7*i:24*7*i+24*5])) # 5 work days
    school_test_work=np.concatenate((school_test_work,school_test[24*7*i:24*7*i+24*5])) 
    

# B: Quantiles Fitting
# Sample
# plt.plot(school_train_work[0:24*14])

quantile=np.linspace(10, 100,num=10)

n1=255 # number of work day in a year
t=np.linspace(1, 24*n1,num=24*n1)

nn_T=1       # T period [number  of days]
# nn_T=20*3  ~ 3 month
w=2*np.pi/(24*nn_T)   # w=f= 2pi/T  T is the time required to finish one wave (step here is hours so the time unit is hour not second)

                      # f=1/T [period/s]  but period =2pi rad   thus  f=1/T [2pi rad /s] = 2pi/T
                      # while T is supposed to be in second it will be hour becasue the time step here is one hour
                      # or w=f=1/T but we take cos(2pi nwt);  here I am taking cos(nwt)
               
       
               
n=100   # number of Foureir terms or degree
    

# cos/sin matrix          
matrix1=np.zeros((len(t),n))
matrix2=np.zeros((len(t),n))
for i in range (n):
    matrix1[:,i]=np.cos((i+1)*w*t);  # % cos matrix
    matrix2[:,i]=np.sin((i+1)*w*t);  # % sin matrix

    
# Fourier quantile regression
demand_quantile=np.zeros((9,24))
for i in range(9):
    T=1-0.1*(i+1) # quantile (beta)
    m=gp.Model()
    muu=m.addVar(vtype='C',lb=-GRB.INFINITY,  name='muu')  
    A=m.addVars(n, lb=-GRB.INFINITY, vtype='C', name='A')
    B=m.addVars(n, lb=-GRB.INFINITY, vtype='C', name='B')
    C=m.addVars(len(t), lb=-GRB.INFINITY, vtype='C', name='C')
    D=m.addVars(len(t), lb=-GRB.INFINITY, vtype='C', name='D')

    # add auxiliary variables for max function
    auxvarpos=m.addVars(len(t), lb=-GRB.INFINITY, vtype='C', name='auxvarpos')
    auxvarneg=m.addVars(len(t), lb=-GRB.INFINITY, vtype='C', name='auxvarneg')
    maxobj1=m.addVars(len(t), lb=0, vtype='C', name="maxobj1")
    maxobj2=m.addVars(len(t), lb=0, vtype='C', name="maxobj2")
    
    # add auxiliary equality constraints
    m.addConstrs((auxvarpos[i] ==  school_train_work[i]-muu-C[i]-D[i]) for i in range(len(t)))
    m.addConstrs((auxvarneg[i] == -school_train_work[i]+muu+C[i]+D[i]) for i in range(len(t)))
    
    # add constraints maxobj1 = max(auxvarpos,0), maxobj2 = max(auxvarneg,0)
    m.addConstrs((maxobj1[i] == gp.max_(auxvarpos[i], constant=0) for i in range(len(t))))
    m.addConstrs((maxobj2[i] == gp.max_(auxvarneg[i], constant=0) for i in range(len(t))))
    
    obj1=gp.quicksum( maxobj1[i] for i in range(len(t)))
    obj2=gp.quicksum( maxobj2[i] for i in range(len(t)))
     
    m.setObjective((T*obj1+(1-T)*obj2)/len(t)) 
    
    # Wrong version:
    # obj1=gp.quicksum(T*np.max((school_train_work[i]-muu-C[i]-D[i]),0)         for i in range(len(t))) 
    # obj2=gp.quicksum((1-T)*np.max(-1*(school_train_work[i]-muu-C[i]-D[i]),0)  for i in range(len(t)))  
    # m.setObjective(obj1+obj2) 
    
    m.addConstrs( C[i]== gp.quicksum(A[k]*matrix1[i][k] for k in range (n)) for i in range (len(t))) 
    m.addConstrs( D[i]== gp.quicksum(B[k]*matrix2[i][k] for k in range (n)) for i in range (len(t))) 
    
    m.optimize()
    
    # Validation of Fourier quantile regression
    plt.plot(t[1:24*n1],school_train_work[1:24*n1],'k-', label='Actual')
    xx=[muu.x+C[i].x+D[i].x for i in range(len(t))] 
    plt.plot(t[1:24*n1],xx[1:24*n1],'r-', label='Fitted')
    plt.legend(loc='upper right')
    plt.xlabel("Time [hrs]")
    plt.ylabel("Demand [kWh]")
    plt.show()
    
    
    n1=7
    plt.plot(t[1:24*n1],school_train_work[1:24*n1],'k-', label='Actual')
    xx=[muu.x+C[i].x+D[i].x for i in range(len(t))] 
    plt.plot(t[1:24*n1],xx[1:24*n1],'r-', label='Fitted')
    plt.legend(loc='upper right')
    plt.xlabel("Time [hrs]")
    plt.ylabel("Demand [kWh]")
    plt.show()
    
    
    n1=nn_T
    plt.plot(t[1:24*n1],school_train_work[1:24*n1],'k-', label='Actual')
    xx=[muu.x+C[i].x+D[i].x for i in range(len(t))] 
    plt.plot(t[1:24*n1],xx[1:24*n1],'r-', label='Fitted')
    plt.legend(loc='upper right')
    plt.xlabel("Time [hrs]")
    plt.ylabel("Demand [kWh]")
    plt.show()


    # Storing the quantile 
    demand_quantile[i,:]=xx[0:24*n1]



# plot demand quantiles
for i in range(9):
    if i==0:
        plt.plot(demand_quantile[i,:],label='0.9')
    else:
        if i==8:
            plt.plot(demand_quantile[i,:],label='0.1')
        else:
            plt.plot(demand_quantile[i,:])
            
plt.xlabel("Time [hrs]")
plt.ylabel("Demand [kWh]")   
plt.legend() 
plt.show()    



# C: Porbabilty Transistion Matrix for demand quantiles
n_day=255
# Determing the quantile at every steps
qunatiles=np.zeros((n_day,24)) # 255 days and 24 hours
for i in range (n_day):
    for j in range (24):
        point=school_train_work[i*24+j]          # every step in the year
        qunatiles_at_point=demand_quantile[:,j]  # qunatiles at that time step of the day
        
        # Find the quantile crossponding to the point
        x=point>qunatiles_at_point
        y=np.where(x==True)
        yy=np.where(x==False)
        if len(yy[0])==9:
            y=9
        else: 
            y=y[0][0]
            if y==0:
                y=1
        quant=1-0.1*y
        
        qunatiles[i,j]=np.round(quant,1)


# Porbabilty Transistion Matrix
qunatiles=np.reshape(qunatiles,(1,n_day*24)) 

def transition_matrix(transitions):
    n = 9 #number of states

    transitions=[ int(transitions[0][i]*10-1) for i in range(len(qunatiles[0])) ] # 0.1 will be 0, 0.2 will be 1, 0.3 will be 2
    M = np.zeros((n,n))   # Porbabilty Transistion Matrix

    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1

    #now convert to probabilities:
    M = M/M.sum(axis=1, keepdims=True)
    return M


M = transition_matrix(qunatiles)   # Porbabilty Transistion Matrix
for row in M: print(' '.join(f'{x:.2f}' for x in row))
mm=np.round(M,2)                   # Porbabilty Transistion Matrix


# Plot the Porbabilty Transistion Matrix
fig, ax = plt.subplots()
min_val, max_val = 0.1, 0.9
intersection_matrix = mm
ax.matshow(intersection_matrix, cmap=plt.cm.Blues)
alpha=['0.1', '0.2','0.3', '0.4','0.5', '0.6','0.7', '0.8','0.9']
ax.set_xticklabels(['']+alpha)
ax.set_yticklabels(['']+alpha)
for i in range(9):
    for j in range(9):
        c = intersection_matrix[j,i]
        ax.text(i, j, str(c), va='center', ha='center', fontsize=10)
plt.show()        
        



# PANs

W=0.01
disc=0
disc_rate=0.001
patience = 10
loss_=0
nnn=2
dist_PANs_all=[]
dist_all=[]

for J1 in [6,8,10]:
    for i1 in [1,2,3,4]:

        # print('Seq-len', J1)
        # print('step', i1)
        print(f'Seq-len: {J1} | Step: {i1}')

        for PANs in [True, False]:
            # Set a fixed random seed
            SEED = 42
            # Set seed for Python's built-in random module
            random.seed(SEED)
            # Set seed for numpy
            np.random.seed(SEED)
            # Set seed for PyTorch
            torch.manual_seed(SEED)
            # ========== PARAMETERS ==========
            SEQ_LEN = J1
            HORIZON = i1
            QUANTILES = 9
            EPOCHS = 200
            BATCH_SIZE = 32*2*2
            HIDDEN_SIZE = 10
            # PANs=False
            
            #  Import School Results
            
            school=pd.read_csv(r'C:\Users\Hussein Sharadga\Desktop\Postdoc Files\UMass\Markov Chain\school.csv')
            school_test=school[329088-96*365+96*4-1:329088-2]# Load profile for one year, starts on January 1, 2010 at 12:00 AM (00:00 in 24 hr style)
            school_test=school_test[0:360*24*4] # The data is recorded every 15 minutes, thus 4*24 points for one day and 360 is assumed to be the 12 months  (We know that 12 months is 365 days)
            school_test=np.array(school_test)
            school_test=np.reshape(school_test,(360*96))
            # Averaging: 15 mins to hourly
            school_test=np.reshape(school_test,(360*24,4))
            school_test=np.mean(school_test,axis=1)         # Hourly demnad for 360 days
            
            # One year for training and one year for test
            x=329088-96*365+96*4-1
            school_train=school[x-96*365-1:x-2]
            school_train=school_train[1*96:-(4)*96+1]
            school_train=np.array(school_train)
            school_train=np.reshape(school_train,(360*96))
            # Averaging: 15 mins to hourly
            school_train=np.reshape(school_train,(360*24,4))
            school_train=np.mean(school_train,axis=1)   
            
            # Work days only: remove the weekends
            school_train_work=[]
            school_test_work=[]
            for i in range(51):
                school_train_work=np.concatenate((school_train_work,school_train[24*7*i:24*7*i+24*5])) # 5 work days
                school_test_work=np.concatenate((school_test_work,school_test[24*7*i:24*7*i+24*5])) 
                
            

            
            def get_quantile_sequence_real(load_series, demand_quantile, n_day=255):
                """
                Maps a time series of load data into quantile levels (0.1, 0.2, ..., 0.9).
                """
                quantiles = np.zeros((n_day, 24))  # shape: (days, hours)
            
                for i in range(n_day):
                    for j in range(24):
                        point = load_series[i * 24 + j]              # load at that time
                        quantiles_at_hour = demand_quantile[:, j]    # all quantile cutoffs for this hour
            
                        # Find the quantile index for the point
                        x = point > quantiles_at_hour
                        y = np.where(x)[0]
                        yy = np.where(~x)[0]
            
                        if len(yy) == 9:
                            q_index = 9
                        else:
                            q_index = y[0] if len(y) > 0 else 0
                            if q_index == 0:
                                q_index = 1
            
                        quant = 1 - 0.1 * q_index
                        quantiles[i, j] = np.round(quant, 1)
                
                # Flatten to 1D sequence of quantiles
                return quantiles.reshape(1, -1)
            
            
            def transition_matrix_real(quantile_seq):
                """
                Computes the transition matrix from a sequence of quantile levels.
                """
                n = 9  # number of quantile states (from 0.1 to 0.9)
                
                # Convert quantile values to integer state indices: 0.1 → 0, 0.2 → 1, ..., 0.9 → 8
                transitions = [int(q * 10 - 1) for q in quantile_seq[0]]
                
                M = np.zeros((n, n))
                for i, j in zip(transitions, transitions[1:]):
                    M[i][j] += 1
            
                # Normalize to row-wise probabilities
                M = M / M.sum(axis=1, keepdims=True)
                return np.round(M, 2)
            
            
            # === Real Data TPM ===
            real_quantile_seq = get_quantile_sequence_real(school_train_work, demand_quantile)
            real_tpm = transition_matrix_real(real_quantile_seq)
            
            # if PANs==True:
            #     print("Real TPM:")
            #     for row in real_tpm:
            #         print(" ".join(f"{x:.2f}" for x in row))
            
            def get_quantile_sequence(load_series, demand_quantile, n_day=BATCH_SIZE):
                """
                Maps a time series of load data into quantile levels (0.1, 0.2, ..., 0.9).
                """
                quantiles = np.zeros(BATCH_SIZE)  # shape: (days, hours)
                j=6
                for i in range(BATCH_SIZE):
                    
                    if j==24:
                        j=0
                    #print(j)    
                    point = load_series[i]              # load at that time
                    quantiles_at_hour = demand_quantile[:, j]    # all quantile cutoffs for this hour
            
                    # Find the quantile index for the point
                    x = point > quantiles_at_hour
                    y = np.where(x)[0]
                    yy = np.where(~x)[0]
            
                    if len(yy) == 9:
                        q_index = 9
                    else:
                        q_index = y[0] if len(y) > 0 else 0
                        if q_index == 0:
                            q_index = 1
            
                    quant = 1 - 0.1 * q_index
                    quantiles[i] = np.round(quant, 1)
                    j=j+1
                
                # Flatten to 1D sequence of quantiles
                return quantiles
            
            
            def transition_matrix(quantile_seq):
                """
                Computes the transition matrix from a sequence of quantile levels.
                """
                n = 9  # number of quantile states (from 0.1 to 0.9)
                
                # Convert quantile values to integer state indices: 0.1 → 0, 0.2 → 1, ..., 0.9 → 8
                transitions = [int(q * 10 - 1) for q in quantile_seq]
                
                M = np.zeros((n, n))
                for i, j in zip(transitions, transitions[1:]):
                    M[i][j] += 1
            
                # Normalize to row-wise probabilities
                #M = M / M.sum(axis=1, keepdims=True)
                row_sums = M.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1  # Prevent divide-by-zero
                M = M / row_sums
                return np.round(M, 2)
            
            
            
            # GANs
            data=school_train_work
            scaler = MinMaxScaler()
            data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
            
            # ========== SEQUENCE CREATION ==========
            def create_sequences(data, seq_len=SEQ_LEN, horizon=HORIZON):
                X, y = [], []
                for i in range(len(data) - seq_len - horizon):
                    X.append(data[i:i+seq_len])
                    y.append(data[i+seq_len:i+seq_len+horizon])
                return np.array(X), np.array(y)
            
            X, y = create_sequences(data)
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)
            #print(y_tensor)
            X_train, X_val, y_train, y_val = train_test_split(
                X_tensor, y_tensor, test_size=0.2, random_state=SEED, shuffle=False
            )
            
            
            data=school_test_work
            data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
            
            def create_sequences(data, seq_len=SEQ_LEN, horizon=HORIZON):
                X, y = [], []
                for i in range(len(data) - seq_len - horizon):
                    X.append(data[i:i+seq_len])
                    y.append(data[i+seq_len])
                return np.array(X), np.array(y)
            
            X_test, y_test = create_sequences(data)
            
            
            # GENERATOR 
            class Generator(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=nnn, batch_first=True)
                    self.fc = nn.Linear(HIDDEN_SIZE, HORIZON)
                
                def forward(self, x):
                    x = x.unsqueeze(-1)  # (B, T) → (B, T, 1)
                    _, (h, _) = self.lstm(x)
                    out = self.fc(h[-1])
                    # print(out.shape)
                    # print('--------')
                    #print(out)
                    return out
                
                
                        
            class Discriminator(nn.Module):
                def __init__(self, quantiles=9):
                    super().__init__()
                    if disc==0:
                        self.conv = nn.Sequential(
                            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # (1, 9, 9) → (32, 9, 9)
                            nn.LeakyReLU(0.2),
                            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (32, 9, 9) → (64, 9, 9)
                            nn.BatchNorm2d(64),
                            nn.LeakyReLU(0.2),
                            nn.Conv2d(64, 128, kernel_size=3, padding=1), # (64, 9, 9) → (128, 9, 9)
                            nn.BatchNorm2d(128),
                            nn.LeakyReLU(0.2),
                            nn.Dropout(0.3),
                
                            nn.Flatten(),                                 # (128, 9, 9) → (10368,)
                            nn.Linear(128 * quantiles * quantiles, 256),  # 10368 → 256
                            nn.LeakyReLU(0.2),
                            nn.Dropout(0.3),
                            nn.Linear(256, 64),                           # 256 → 64
                            nn.LeakyReLU(0.2),
                            nn.Linear(64, 1),                             # 64 → 1
                            nn.Sigmoid()                                  # Output between 0 and 1
                            )
                    else:
                        self.conv = nn.Sequential(
                            nn.Conv2d(1, 16, kernel_size=3, padding=1),
                            nn.LeakyReLU(0.2),
                            nn.Conv2d(16, 32, kernel_size=3, padding=1),
                            nn.LeakyReLU(0.2),
                            nn.Flatten(),
                            nn.Linear(32 * quantiles * quantiles, 64),
                            nn.LeakyReLU(0.2),
                            nn.Linear(64, 1),
                            nn.Sigmoid()
                            )
                
                def forward(self, x):
                    return self.conv(x.unsqueeze(1))  # Input (batch, 9, 9) → (batch, 1, 9, 9)
                        
            
            # ========== TRAINING ==========
            generator = Generator()
            if PANs==True:
                discriminator = Discriminator()
            
            g_optimizer = optim.Adam(generator.parameters(), lr=0.001)
            if PANs==True:
                d_optimizer = optim.Adam(discriminator.parameters(), lr=disc_rate)
            

            bce_loss = nn.BCELoss()
            
            mse_loss = nn.MSELoss()
            
            real_label = 0.9
            fake_label = 0.0
            
            # Training Loop
            best_val_loss = np.inf
            best_model_state = copy.deepcopy(generator.state_dict())
            
            counter = 0
            #loss_epoch_val = []
            #loss_epoch_train=[]
            
            for epoch in range(EPOCHS):
                generator.train()
                if PANs==True:
                    discriminator.train()
                
                perm = torch.randperm(X_tensor.size(0))
                for i in range(0, X_tensor.size(0), BATCH_SIZE):
                    idx = perm[i:i+BATCH_SIZE]
                    x_batch = X_tensor[idx]
                    y_batch = y_tensor[idx]

                    # --- Generator forward ---
                    g_output = generator(x_batch)
                    # Forecast loss
                    loss_forecast = mse_loss(g_output, y_batch)
            
                    # Compute fake TPM
                    fake_preds = g_output.detach().squeeze().numpy()
                    fake_preds = fake_preds.reshape(-1, 1)  # shape will be (384, 1)
                    past_vals = x_batch[:, -1].detach().numpy()
                    if fake_preds.shape!=(BATCH_SIZE*HORIZON,1):
                        #print('hit!')
                        continue
                    
                    if PANs==True:
                        restored_fake_preds = scaler.inverse_transform(np.array(fake_preds)).flatten()
                        fake_quant_seq = get_quantile_sequence(restored_fake_preds, demand_quantile)
                        fake_tpm = transition_matrix(fake_quant_seq)
                        fake_tpm_tensor = torch.tensor(fake_tpm, dtype=torch.float32).unsqueeze(0)
                
                        # === Discriminator ===
                        discriminator.zero_grad()
                        real_tpm_tensor = torch.tensor(real_tpm, dtype=torch.float32).unsqueeze(0)
                
                        d_real = discriminator(real_tpm_tensor)
                        d_fake = discriminator(fake_tpm_tensor.detach())
                
                        d_loss_real = bce_loss(d_real, torch.ones_like(d_real))
                        d_loss_fake = bce_loss(d_fake, torch.zeros_like(d_fake))
                        d_loss = d_loss_real + d_loss_fake
                        d_loss.backward()
                        d_optimizer.step()
            
                    # === Generator ===
                    generator.zero_grad()
                    if PANs==True:
                        d_fake_for_g = discriminator(fake_tpm_tensor)
                        g_loss_adv = bce_loss(d_fake_for_g, torch.ones_like(d_fake_for_g))
                        g_total_loss = loss_forecast + W* g_loss_adv
                    else:
                        g_total_loss = loss_forecast
                        
                    g_total_loss.backward()
                    g_optimizer.step()
                    
                    
                    
                    # === VALIDATION LOSS ===
                    generator.eval()
                    with torch.no_grad():
                        val_preds = generator(X_val)
                        val_loss = mse_loss(val_preds, y_val).item()  # <-- Validation MSE
                        if loss_==1 and PANs==True:
                            # Compute fake TPM
                            fake_preds = val_preds.detach().squeeze().numpy()
                            fake_preds = fake_preds.reshape(-1, 1)  # shape will be (384, 1)
                          
                            restored_fake_preds = scaler.inverse_transform(np.array(fake_preds)).flatten()
                            fake_quant_seq = get_quantile_sequence(restored_fake_preds, demand_quantile)
                            fake_tpm = transition_matrix(fake_quant_seq)
                            fake_tpm_tensor = torch.tensor(fake_tpm, dtype=torch.float32).unsqueeze(0)
                            d_fake_for_g = discriminator(fake_tpm_tensor)
                            g_loss_adv = bce_loss(d_fake_for_g, torch.ones_like(d_fake_for_g))
                            val_loss = val_loss + W* g_loss_adv
                    


                    # loss_epoch_val.append(val_loss)
                    # loss_epoch_train.append(g_total_loss.item())
                    
                    # === EARLY STOPPING LOGIC BASED ON VALIDATION LOSS ===
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = copy.deepcopy(generator.state_dict())
                        counter = 0
                    else:
                        counter += 1
                        if counter >= patience:
                            #print(f"Early stopping triggered at epoch {epoch}")
                            break

                    # # === PROGRESS PRINTOUT ===
                    # if epoch % 10 == 0:
                    #     print(f"Epoch {epoch}/{EPOCHS} | Val Loss: {val_loss:.4f} | D Loss: {d_loss.item():.4f}")
                    #     print("Fake TPM:")
                    #     for row in fake_tpm:
                    #         print(" ".join(f"{x:.2f}" for x in row))

            # Restore best generator
            generator.load_state_dict(best_model_state)
            #print(f"Best model restored with Val Loss: {best_val_loss:.4f}")

            
                # if epoch % 10 == 0:
                #     if PANs==True:
                #         print(f"Epoch {epoch}/{EPOCHS} | G Loss: {g_total_loss.item():.4f} | D Loss: {d_loss.item():.4f}")
                #         if epoch % 10 == 0:
                #             print("Fake TPM:")
                #             for row in fake_tpm:
                #                 print(" ".join(f"{x:.2f}" for x in row))
                #     else:
                #         print(f"Epoch {epoch}/{EPOCHS} | G Loss: {g_total_loss.item():.4f} | D Loss: {0}")
            
                #         # fig, ax = plt.subplots()
                #         # min_val, max_val = 0.1, 0.9
                #         # intersection_matrix = fake_tpm
                #         # ax.matshow(intersection_matrix, cmap=plt.cm.Blues)
                #         # alpha=['0.1', '0.2','0.3', '0.4','0.5', '0.6','0.7', '0.8','0.9']
                #         # ax.set_xticklabels(['']+alpha)
                #         # ax.set_yticklabels(['']+alpha)
                #         # for i in range(9):
                #         #     for j in range(9):
                #         #         c = intersection_matrix[j,i]
                #         #         ax.text(i, j, str(c), va='center', ha='center', fontsize=10)
                                  
            
            
            def predict_next_step(generator, input_sequence):
                """
                input_sequence: 1D array of length = input_size (e.g., 6)
                """
                generator.eval()
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0)  # shape (1, input_size)
                    prediction = generator(input_tensor)
                    #print(prediction[0])
                    if HORIZON>1:
                        return list(prediction.squeeze().numpy())  # shape (3,)
                    else:
                        return prediction.numpy()  # shape (3,)
            
            
            def mean_squared_error(x, y):
                if len(x) != len(y):
                    raise ValueError("Lists x and y must have the same length.")
                return sum((xi - yi) ** 2 for xi, yi in zip(x, y)) / len(x)
            
            
            # all
            XX_all=[]
            YY_all=[]
            for i in range(len(X_test)-HORIZON+1):           
                prediction=predict_next_step(generator, X_test[i])
            
                if HORIZON>1:
                    restored_prediction = scaler.inverse_transform(np.array([prediction])).flatten()
                    XX_all.extend(list(restored_prediction))
                    
                    restored_actual = scaler.inverse_transform(np.array([y_test[i:i+HORIZON]])).flatten()
                    YY_all.extend(list(restored_actual))
                else:
                    restored_prediction = scaler.inverse_transform(np.array(prediction)).flatten()
                    XX_all.append(restored_prediction[0])
                    
                    restored_actual = scaler.inverse_transform(np.array([[y_test[i]]])).flatten()
                    YY_all.append(restored_actual[0])
                    

            
            def mean_squared_error(x, y):
                if len(x) != len(y):
                    raise ValueError("Lists x and y must have the same length.")
                return sum((xi - yi) ** 2 for xi, yi in zip(x, y)) / len(x)
            
            if PANs:
                PANs_MSE=mean_squared_error(XX_all,YY_all)
                # print('PANs-MSE = ',PANs_MSE) 
                m_PANs=transition_matrix(get_quantile_sequence(XX_all, demand_quantile))
       
            else:
                NN_MSE=mean_squared_error(XX_all,YY_all)
                # print('NN-MSE = ', NN_MSE)      
                m=transition_matrix(get_quantile_sequence(XX_all, demand_quantile))     
                                
            
            if PANs==False:    
                print((NN_MSE-PANs_MSE)/NN_MSE*100)
                # Calculate distances
                dist_PANs = frobenius_norm(m_PANs, real_tpm)
                dist_= frobenius_norm(m, real_tpm)

                dist_PANs_all.append(dist_PANs)
                dist_all.append(dist_)
                
plt.plot(dist_all,'ro')                      
plt.plot(dist_PANs_all,'kx')
plt.show()               
