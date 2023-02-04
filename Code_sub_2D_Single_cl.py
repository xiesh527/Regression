from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import Rbf
np.set_printoptions(suppress=False)
from sklearn.neighbors import KNeighborsRegressor
from scipy.spatial import distance

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
            per_phi = np.exp(-(alpha * np.linalg.norm(data_row - rbf[j])) ** 2) * np.exp(np.exp(-(alpha1 * np.linalg.norm(data_row - rbf[j]))) - 1)
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

#
trial_time = 1000
#
#
SNR_base_db = -21
rbf_ratio = 0.08

data_train = genvec(2,1000)
plt.plot(data_train[:,0], data_train[:,1], 'o')
plt.show()

mu1 = [0, 0]
sigma1 = [50, 50]
sigma1_x1 = sigma1[0]
sigma1_x2 = sigma1[1]
print(sigma1_x1)
print(sigma1_x2)
sigma1 = np.diag(sigma1)

data_train_cl1 = [0, 0]

for i in range(len(data_train)):
    data_train_cl1 = np.vstack([data_train_cl1, gencl_row(data_train[i:i+1], mu1, sigma1)])

data_train_cl1 = data_train_cl1[1:len(data_train_cl1)]
num_rbf = rbf_ratio*len(data_train_cl1)
num_rbf_axis = int(np.round(np.sqrt(num_rbf)))
rbf_cl1_x1 = np.linspace(mu1[0] - 2*sigma1_x1, mu1[0] + 2*sigma1_x1, num = num_rbf_axis)
rbf_cl1_x2 = np.linspace(mu1[1] - 2*sigma1_x2, mu1[1] + 2*sigma1_x2, num = num_rbf_axis)

rbf_cl1 = [0, 0]

for i in range(num_rbf_axis):
    for j in range(num_rbf_axis):
        rbf_row = np.append(rbf_cl1_x1[i], rbf_cl1_x2[j])
        rbf_cl1 = np.vstack([rbf_cl1, rbf_row])

rbf_cl1 = rbf_cl1[1:len(rbf_cl1)]
print(np.shape(rbf_cl1))

print(num_rbf_axis)
print(data_train_cl1)
plt.plot(data_train_cl1[:,0], data_train_cl1[:,1], 'o')
plt.plot(rbf_cl1[:,0], rbf_cl1[:,1], 'x')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Reference points distribution within a cluster')
plt.show()

beta = np.linspace(0.1, 1.2, 20)
alpha_base = np.sqrt(0.35**2+0.35**2)
alpha1 = 10

cond_list = []
for k in range(len(beta)):

    alpha = alpha_base*beta[k]

    cond_list_all_j = []
    for j in range(1000):
        data_train = genvec(2, 1000)
        data_train_cl1 = [0, 0]

        for i in range(len(data_train)):
            data_train_cl1 = np.vstack([data_train_cl1, gencl_row(data_train[i:i + 1], mu1, sigma1)])

        phi_buf = synth_phi_mat(data_train_cl1, rbf_cl1, alpha, alpha1)
        IM_buf = get_IM(phi_buf, rbf_cl1)
        cond_list_all_j = np.append(cond_list_all_j, np.linalg.cond(IM_buf))
        print(j, np.linalg.cond(IM_buf))
    cond_per_i = np.mean(cond_list_all_j)
    print(cond_per_i)
    cond_list = np.append(cond_list, cond_per_i)


cond_list_log = np.log10(cond_list)
plt.plot(beta, cond_list_log, 'o')
plt.xlabel('Alpha tuner: beta')
plt.ylabel('Logarithm of condition number')
plt.title('Empirical shape parameter tuning process in 2D')
plt.show()

