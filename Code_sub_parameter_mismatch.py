from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import Rbf
np.set_printoptions(suppress=False)

def genvec(dim,num):
    data = np.random.randn(num, dim)
    return data

def genmu(val, vec):
    mu = [val for i in range(len(vec))]
    return mu

def gencl(cl, mu, sigmas):
    result = np.dot(cl, sigmas) + mu
    return result

def synthesis_cluster(data, rfd, alpha):
    psi = np.zeros((1, len(rfd)))
    for i in range(len(data)):

        psi_per_row = []
        psi_per_row = np.asarray(psi_per_row)
        for j in range(len(rfd)):
            row_per_basis = np.exp(-(alpha * np.abs(data[i] - rfd[j])) ** 2)

            psi_per_row = np.append(psi_per_row, row_per_basis)
            print(np.shape(psi_per_row))
        psi = np.vstack([psi, psi_per_row])

    psi = psi[1:len(psi)]
    return psi

def get_y(psi, q):
    y = []
    y = np.asarray(y)
    for i in range(len(psi)):
        y_per_row = np.dot(psi[i], q.T)
        y = np.append(y, y_per_row)
    return y

def get_vnaa(EIM, psi):
    vnaa = EIM[:, 1:len(psi)]
    vnaa = vnaa[1:len(psi.T)]
    return vnaa

def get_vna1(EIM):
    vna1 = EIM[:, 0]
    vna1 = vna1[1:len(vna1)]
    return vna1

data_test = genvec(1, 250)



mu1 = [0]
sigma1 = [50]
sigma1 = np.diag(sigma1)
tag_list = []

ratio1 = 0.1
rbf_cl1 = [-100, -50, 0, 50, 100]

delta = 100/(50-1)

beta = 1.6e-3*50
alpha = beta*(1/delta)
alpha1 = 0

gamma = 1
alpha_analysis = gamma*alpha
#synthesis model
for k in range(1):
    for i in range(len(data_test)):
        data_test[i:i + 1] = gencl(data_test[i:i + 1], mu1, sigma1)

    phi_test = np.zeros((1, 250))
    phi_test_ana = np.zeros((1, 250))
    for i in range(len(rbf_cl1)):
        phi_i_test = np.exp(-(alpha * np.abs(data_test - rbf_cl1[i])) ** 2) * np.exp(np.exp(-(alpha1 * np.abs(data_test - rbf_cl1[i]))) - 1)
        phi_i_test_ana = np.exp(-(alpha_analysis * np.abs(data_test - rbf_cl1[i])) ** 2) * np.exp(np.exp(-(alpha1 * np.abs(data_test - rbf_cl1[i]))) - 1)
        phi_i_test = phi_i_test.T

        phi_i_test_ana = phi_i_test_ana.T
        phi_test = np.vstack([phi_test, phi_i_test])
        phi_test_ana = np.vstack([phi_test_ana, phi_i_test_ana])
    print(np.shape(phi_test_ana))
    phi_test = phi_test[1:len(phi_test)]
    phi_test = phi_test.T
    phi_test_ana = phi_test_ana[1:len(phi_test_ana)]
    phi_test_ana = phi_test_ana.T


for j in range(1):

    data = genvec(1, 1000)
    for i in range(len(data)):
        data[i:i + 1] = gencl(data[i:i + 1], mu1, sigma1)
        # print(data[i:i+1])
        #tag_list = np.append(tag_list, 1)
    phi = np.zeros((1,1000))
    phi_analysis = np.zeros((1, 1000))
    for i in range(len(rbf_cl1)):
        phi_i = np.exp(-(alpha*np.abs(data-rbf_cl1[i]))**2)*np.exp(np.exp(-(alpha1*np.abs(data-rbf_cl1[i])))-1)

        phi_i_analysis = np.exp(-(alpha_analysis * np.abs(data - rbf_cl1[i])) ** 2) * np.exp(np.exp(-(alpha1 * np.abs(data - rbf_cl1[i]))) - 1)

        phi_i = phi_i.T
        phi_i_analysis = phi_i_analysis.T
        phi = np.vstack([phi, phi_i])
        phi_analysis = np.vstack([phi_analysis, phi_i_analysis])

    phi = phi[1:len(phi)]
    phi = phi.T
    phi_analysis = phi_analysis[1:len(phi_analysis)]
    phi_analysis = phi_analysis.T

    a = [1, 1, 1, 1, 1]
    a = np.asarray(a)
    y_signal = []
    y_signal_test = []
    for i in range(len(phi)):

        y = np.dot(phi[i], a.T)

        y_signal = np.append(y_signal, y)

    psiT = np.vstack([y_signal, phi.T])

    for i in range(len(phi_test)):
        y_test = np.dot(phi_test[i], a.T)
        y_signal_test = np.append(y_signal_test, y_test)

psi = psiT.T

#plt.plot(data, psi[:, 1], 'o')
#plt.plot(data, psi[:, 2], 'o')
plt.plot(data, psi[:, 3], 'o')
#plt.plot(data, psi[:, 4], 'o')
#plt.plot(data, psi[:, 5], 'o')
plt.show()

power_y = y_signal**2

avg_pow_y = np.mean(power_y)
avg_pow_y_db = 10*np.log10(avg_pow_y)
mean_noise = 0
perform_list = []
print(phi_test-phi_test_ana)
for i in range(1000):
    # ----------------
    SNR_db = 20
    # ----------------
    noise_db = avg_pow_y_db - SNR_db
    noise_watts = 10 ** (noise_db / 10)
    noise = np.random.normal(mean_noise, np.sqrt(noise_watts), len(data))
    noise_y = y_signal + noise

    psi_analysisT = np.vstack([noise_y, phi_analysis.T])
    psi_analysis = psi_analysisT.T

    EIM = np.dot(psi_analysisT, psi_analysis)
    vna1_ana = get_vna1(EIM)
    vnaa_ana = get_vnaa(EIM, psi_analysis)

    a_hat = np.dot(np.linalg.inv(vnaa_ana), vna1_ana)
    print(np.linalg.cond(vnaa_ana))
    print(a_hat)
    y_pred = []

    for j in range(len(phi_test_ana)):
        y_i_pred = np.dot(phi_test_ana[j], a_hat.T)
        y_pred = np.append(y_pred, y_i_pred)

    diff = np.abs(y_pred-y_signal_test)

    diff_pertrial = np.mean(diff)
    perform_list = np.append(perform_list, diff_pertrial)
print(alpha_analysis)
print(np.mean(perform_list))

