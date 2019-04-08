import pandas as pd
import numpy as np
from math import log
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import pykalman as pykal
import pickle

datos = "base.pkl"
futures = pickle.load(open(datos, "rb"))

futures.iloc[:,0]
spot = futures.iloc[:,0]
log_spot = np.log(spot)
log_spot.shape
my_tuple = pd.DataFrame(futures).shape
my_tuple
num_contracts = my_tuple[1]
num_days = my_tuple[0]

log_futures = np.log(futures)



k=1.49
sigma_short = 0.286
sigma_long = 0.145
rho = 0.3
Lambda =  0.157
mu =0.0115

num_contracts
months = np.arange(num_contracts)
months
days = 30/360
Tau = np.array(months*days)
i = 0
factor_descuento = np.zeros(num_contracts)
Tau

while i < num_contracts:
    for t in Tau:
        factor_descuento[i] = np.exp(-t)
        i += 1


C = np.ones (num_contracts)
obs = np.vstack((factor_descuento,C))

d = np.zeros(num_contracts)
j = 0
for mat in Tau:
    d[j] = mu*mat+ (1-np.exp(-2*k*mat))*(Lambda/k) + 0.5*((1-np.exp(-2*k*mat))*((sigma_short*sigma_short)/(2*k))+(sigma_long*sigma_long)*mat) + (1-np.exp(-k*mat)*rho*sigma_long*sigma_short/k)
    j += 1
d
d.shape = (num_contracts,)


x_0 = log_spot.iloc[0]/2
epsilon_0 = log_spot.iloc[0]/2
initial_mean = np.array([[x_0],[epsilon_0]])
tiempo = np.arange(num_days)
t = np.arange(num_days)
delta_t = 1/len(t)
tiempo
delta_t
W = []
for t in tiempo:
    W.append(np.array([[(1-np.exp(-2*k*t))*(Lambda/k),(1-np.exp(-k*t)*rho*sigma_long*sigma_short/k)],[(1-np.exp(-k*t)*rho*sigma_long*sigma_short/k), sigma_long*sigma_long*t]]))
    W[t].shape = (2,2)
type(W[0])
# V= covarianza de las observaciones
V = np.identity(2)

#matriz de transicion
transition_M= np.array([[np.exp(-k*delta_t), 0], [0,1]])
measurements = log_futures
measurements.head()
measurements.iloc[:,0]

d.shape = (7,)

kf_2= KalmanFilter(n_dim_obs = num_contracts,
n_dim_state=2,
transition_covariance = np.eye(2),
transition_matrices= transition_M,
observation_matrices = obs.T,
observation_offsets = d,
em_vars= ["observation_covariance", "transition_offsets", "initial_state_mean", "initial_state_covariance"])
kf_2.filter(measurements)

state_means, state_covs = kf_2.filter(measurements)
t = np.arange(num_days)
t2 = np.arange(log_spot.shape[0])


lista = []


for elem in range(len(d)):
    lista.append(state_means[t,0]+ state_means[t,1] + d[elem])


np.save("cov_estados_crudo_SM.npy", state_covs)
np.save("state_means_SM.npy",state_means)




for iter in range(num_contracts):
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.plot(t, state_means[t,0]+ state_means[t,1]+d[iter])
    plt.plot(t2, measurements.iloc[:,iter])

    plt.legend(["Estimador filtrado", "F_" + str(iter)])
    plt.xlabel('Tiempo')
    plt.ylabel('Log Precio')
    plt.savefig("oro"+str(iter)+".png" )




fig, ax = plt.subplots(figsize=(15, 9))

plt.plot(t, state_means[t,0]+ state_means[t,1]+d[0])
plt.plot(t2, measurements.iloc[:,0])
plt.legend(["Estimador filtrado", "F_1"])
plt.xlabel('Tiempo')
plt.ylabel('Log Precio')

plt.plot(t, state_means[t,0]+ state_means[t,1]+d[1])
plt.plot(t2, measurements.iloc[:,1])

plt.legend(["Estimador filtrado", "F_2"])
plt.xlabel('Tiempo')
plt.ylabel('Log Precio')

plt.plot(t, state_means[t,0]+ state_means[t,1]+d[2])
plt.plot(t2, measurements.iloc[:,2])

plt.legend(["Estimador filtrado", "F_3"])
plt.xlabel('Tiempo')
plt.ylabel('Log Precio')

plt.plot(t, state_means[t,0]+ state_means[t,1]+d[3])
plt.plot(t2, measurements.iloc[:,3])

plt.legend(["Estimador filtrado", "F_4"])
plt.xlabel('Tiempo')
plt.ylabel('Log Precio')

plt.plot(t, state_means[t,0]+ state_means[t,1]+d[4])
plt.plot(t2, measurements.iloc[:,4])

plt.legend(["Estimador filtrado", "F_5"])
plt.xlabel('Tiempo')
plt.ylabel('Log Precio')

plt.plot(t, state_means[t,0]+ state_means[t,1]+d[5])
plt.plot(t2, measurements.iloc[:,5])

plt.legend(["Estimador filtrado", "F_6"])
plt.xlabel('Tiempo')
plt.ylabel('Log Precio')

plt.plot(t, state_means[t,0]+ state_means[t,1]+d[6])
plt.plot(t2, measurements.iloc[:,6])

plt.legend(["Estimador filtrado", "F_7"])
plt.xlabel('Tiempo')
plt.ylabel('Log Precio')
