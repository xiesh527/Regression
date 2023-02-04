from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import Rbf
np.set_printoptions(suppress=False)
from sklearn.neighbors import KNeighborsRegressor

def genvec(dim,num):
    data = np.random.randn(num, dim)
    return data

def genmu(val, vec):
    mu = [val for i in range(len(vec))]
    return mu

def gencl_row(cl, mu, sigmas):
    result = np.dot(cl, sigmas) + mu
    return result


def synth_phi_mat(data, rbf, alpha, alpha1):
    phi = np.zeros((1, len(rbf)))
    for i in range(len(data)):
        data_row = data[i]
        phi_buf = []
        for j in range(len(rbf)):
            per_phi = np.exp(-(alpha * np.abs(data_row - rbf[j])) ** 2) * np.exp(np.exp(-(alpha1 * np.abs(data_row - rbf[j]))) - 1)
            phi_buf = np.append(phi_buf, per_phi)
        phi = np.vstack([phi, phi_buf])
    phi = phi[1:len(phi)]
    return(phi)


def get_ahat(IM, phi, y):
    inter = np.dot(np.linalg.inv(IM), phi.T)
    a_hat = np.dot(inter, y)
    return a_hat

def get_y(phi, a):
    y = []
    for i in range(len(phi)):
        signal_buf = np.dot(phi[i], a.T)
        y = np.append(y, signal_buf)
    return y

def get_IM(phi, rbf):
    IM = np.zeros((len(rbf), len(rbf)))
    for i in range(len(phi)):
        per_IM = np.outer(phi[i].T, phi[i])
        IM = IM + per_IM
    return IM

def get_vnaa(EIM, psi):
    vnaa = EIM[:, 1:len(psi)]
    vnaa = vnaa[1:len(psi.T)]
    return vnaa

def get_vna1(EIM):
    vna1 = EIM[:, 0]
    vna1 = vna1[1:len(vna1)]
    return vna1

data_train = genvec(1, 550)
data_test = genvec(1,250)
#-------
rbf_ratio = 0.08
#-------
mu1 = [500]
sigma1 = [75]
sigma1 = np.diag(sigma1)
pro_cl1 = 0.55
data_train_cl1 = []
data_test_cl1 = []
tag_list_cl1 = []



for i in range(len(data_train)):
    data_train_cl1 = np.append(data_train_cl1, gencl_row(data_train[i:i + 1], mu1, sigma1))

rbf_cl1 = np.linspace(mu1 - 2*sigma1, mu1 + 2*sigma1, num = int(rbf_ratio*len(data_train_cl1)))
a_cl1 = np.zeros((1, len(rbf_cl1))) + 1


gamma = 1

#beta = 1.6e-3*50
#alpha_cl1 = 1#beta*(1/delta_cl1)
alpha1 = 10
alpha = np.linspace(1e-1, 1, 100)
alpha_s = 1
phi_train = synth_phi_mat(data_train_cl1, rbf_cl1, alpha_s, alpha1)
IM_1 = get_IM(phi_train, rbf_cl1)
print(np.linalg.cond(IM_1))
cond_list = []

cond_list_log = []
for i in range(len(alpha)):
    alpha_buf = alpha[i]
    avg_log_cond = []
    for j in range(1000):
        data = genvec(1, 450)
        print(j)
        data_train_99 = gencl_row(data, mu1, sigma1)
        phi_t = synth_phi_mat(data_train_99, rbf_cl1, alpha_buf, alpha1)
        IM_t = get_IM(phi_t, rbf_cl1)
        cond_buf = np.linalg.cond(IM_t)
        cond_log_buf = np.log10(cond_buf)
        avg_log_cond = np.append(avg_log_cond, cond_log_buf)
    avg_log_at_i = np.mean(avg_log_cond)
    cond_list_log = np.append(cond_list_log,  avg_log_at_i)
    print(avg_log_at_i)




plt.plot(alpha, cond_list_log, 'o')
plt.ylabel('log10(Var_K)')
plt.xlabel('Alpha_ana')
plt.show()

